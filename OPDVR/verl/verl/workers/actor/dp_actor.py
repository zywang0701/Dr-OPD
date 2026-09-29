# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.workers.actor.credit_gate import _all_reduce_sum
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False, top_k=0, student_top_k_ids=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            topk_ids: # (bs, response_len, k)
            topk_log_probs: # (bs, response_len, k)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16,
                            enabled=not getattr(self, "fp32_forward", False)):   # E030: fp32 FD copy
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            topk_ids = None
            topk_log_probs = None
            
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating
                
                need_logits = top_k > 0

                if self.use_fused_kernels and not need_logits:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    
                    # Optimization: when top_k > 0, compute log_softmax once and gather both
                    # log_probs and topk_log_probs to avoid duplicate computation and gradient
                    # issues from inplace operations
                    need_topk = top_k > 0
                    if need_topk:
                        # Compute log_softmax once for both target and topk tokens
                        # Note: we don't use inplace_backward here to ensure correct gradients
                        # when both log_probs and topk_log_probs are needed
                        log_probs_all = torch.log_softmax(logits_rmpad, dim=-1)
                        # Gather log_probs for target tokens
                        log_probs = log_probs_all.gather(
                            dim=-1, index=input_ids_rmpad_rolled.unsqueeze(-1)
                        ).squeeze(-1)
                    else:
                        log_probs = logprobs_from_logits(
                            logits=logits_rmpad,
                            labels=input_ids_rmpad_rolled,
                            inplace_backward=inplace_backward,
                        )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )
                    
                    if need_topk:
                        if student_top_k_ids is not None:
                             # Use specific IDs (from rollout)
                             topk_ids = student_top_k_ids
                             if student_top_k_ids.ndim == 3: # (bsz, seqlen, k)
                                 # We are in rmpad mode, but student_top_k_ids is padded 3D tensor
                                 # We need to extract the relevant tokens aligning with input_ids_rmpad_rolled
                                 
                                 # This is tricky because student_top_k_ids is shaped (batch, seq, k)
                                 # and logits_rmpad is (total_nnz, vocab)
                                 # We need to flatten student_top_k_ids to (total_nnz, k) using indices
                                 
                                 # Re-use the indices computed from unpad_input
                                 # indices: (total_nnz,) 
                                 # student_top_k_ids: (batch, seq, k)
                                 
                                 # 1. If student_top_k_ids only covers the response, pad it to match full sequence length
                                 if student_top_k_ids.shape[1] != seqlen:
                                     full_student_top_k_ids = torch.zeros((batch_size, seqlen, top_k), 
                                                                         dtype=student_top_k_ids.dtype, 
                                                                         device=student_top_k_ids.device)
                                     full_student_top_k_ids[:, -response_length-1:-1, :] = student_top_k_ids
                                     student_top_k_ids = full_student_top_k_ids

                                 # 2. Flatten student_top_k_ids to (batch*seq, k)
                                 flat_ids = student_top_k_ids.view(-1, top_k)
                                 
                                 # 3. Select using indices
                                 # Note: indices are from attention_mask, which aligns with how logits_rmpad represents data
                                 topk_ids_rmpad = flat_ids[indices] # (total_nnz, k)
                                 
                                 # If 'student_top_k_ids' in batch has shape (batch, seq_len, k), then:
                                 topk_ids = topk_ids_rmpad
                                 
                             else:
                                 # If it's already flattened? Unlikely.
                                 pass

                        else:
                             # Legacy/Resample behavior
                             _, topk_ids = torch.topk(logits_rmpad, k=top_k, dim=-1)

                        # Use pre-computed log_probs_all (always available when need_topk=True)
                        topk_log_probs = log_probs_all.gather(dim=-1, index=topk_ids)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if top_k > 0:
                         topk_ids = gather_outputs_and_unpad(
                            topk_ids,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                         )
                         topk_log_probs = gather_outputs_and_unpad(
                            topk_log_probs,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                         )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                
                if top_k > 0:
                    full_topk_ids = pad_input(
                        hidden_states=topk_ids,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    full_topk_log_probs = pad_input(
                        hidden_states=topk_log_probs,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                
                if top_k > 0:
                    topk_ids = full_topk_ids[:, -response_length - 1 : -1, :]
                    topk_log_probs = full_topk_log_probs[:, -response_length - 1 : -1, :]

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating
                
                need_logits = top_k > 0
                if self.use_fused_kernels and not need_logits:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    
                    # Optimization: when top_k > 0, compute log_softmax once and gather both
                    # log_probs and topk_log_probs to avoid duplicate computation
                    need_topk = top_k > 0
                    if need_topk:
                        # Compute log_softmax once for both target and topk tokens
                        log_probs_all = torch.log_softmax(logits, dim=-1)
                        # Gather log_probs for target tokens (responses)
                        log_probs = log_probs_all.gather(
                            dim=-1, index=micro_batch["responses"].unsqueeze(-1)
                        ).squeeze(-1)
                    else:
                        log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    
                    if need_topk:
                        if student_top_k_ids is not None:
                             topk_ids = student_top_k_ids
                             # Ensure shape alignment if needed, but for non-rmpad (bsz, seq, k) should match logits (bsz, seq, vocab) dim 0,1
                        else:
                             _, topk_ids = torch.topk(logits, k=top_k, dim=-1)
                        
                        # Use pre-computed log_probs_all (always available when need_topk=True)
                        topk_log_probs = log_probs_all.gather(dim=-1, index=topk_ids)

            return entropy, log_probs, topk_ids, topk_log_probs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_probs_for_ids(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability for specific token ids
        Args:
            data (DataProto): a DataProto containing input_ids, attention_mask, position_ids, responses, 
                             and target_ids (batch, response_len, k) in batch
        Returns:
            torch.Tensor: (batch, response_len, k) log probs for target_ids
        """
        # set to eval
        self.actor_module.eval()

        target_ids = data.batch["target_ids"]
        
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "target_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        
        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        topk_log_probs_lst = []
        top_k = target_ids.shape[-1]

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            mb_target_ids = model_inputs["target_ids"]
            with torch.no_grad():
                # We reuse _forward_micro_batch. It returns (entropy, log_probs, topk_ids, topk_log_probs)
                _, _, _, topk_log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=False, 
                    top_k=top_k, student_top_k_ids=mb_target_ids
                )
            # Keep on GPU to avoid expensive CPU-GPU transfer for large top-k
            # topk_log_probs = topk_log_probs.to("cpu")
            topk_log_probs_lst.append(topk_log_probs)

        topk_log_probs_tensor = torch.concat(topk_log_probs_lst, dim=0)

        if use_dynamic_bsz:
            topk_log_probs_tensor = restore_dynamic_batch(topk_log_probs_tensor, batch_idx_list)

        return topk_log_probs_tensor

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_distillation_reward(self, data: DataProto) -> DataProto:
        """Compute the distillation reward (rm_scores) on GPU
        Args:
            data (DataProto): containing all necessary tensors for distillation reward calculation
        Returns:
            DataProto: containing rm_scores and other updated tensors (e.g., union_ids)
        """
        # Set to eval mode for forward passes
        self.actor_module.eval()

        # 1. Extract parameters from meta_info
        top_k = data.meta_info.get("log_prob_top_k", 0)
        strategy = data.meta_info.get("top_k_strategy", "only_stu")
        kl_estimator = data.meta_info.get("kl_estimator", "k1")
        reward_weight_mode = data.meta_info.get("reward_weight_mode", "student_p")  # "student_p", "teacher_p", or "none"
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        # 2. Compute Student Log Probs on Teacher IDs if needed
        # (This replaces the previous call to compute_log_probs_for_ids in ray_trainer)
        S_on_T = None
        if strategy in ["only_tch", "intersection", "union", "union-intersection"]:
            target_ids = data.batch["teacher_top_k_ids"]
            
            # Select keys for micro-batching
            has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
            non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
            
            # We need to pass target_ids to _forward_micro_batch, but since we are micro-batching, 
            # we should split target_ids as well.
            mb_data = data.select(batch_keys=select_keys + ["teacher_top_k_ids"], 
                                 non_tensor_batch_keys=non_tensor_select_keys)
            
            if use_dynamic_bsz:
                max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                micro_batches, batch_idx_list = prepare_dynamic_batch(mb_data, max_token_len=max_token_len)
            else:
                micro_batches = mb_data.split(micro_batch_size)

            S_on_T_lst = []
            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                mb_target_ids = model_inputs["teacher_top_k_ids"]
                with torch.no_grad():
                    _, _, _, topk_log_probs = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=False, 
                        top_k=top_k, student_top_k_ids=mb_target_ids
                    )
                S_on_T_lst.append(topk_log_probs)

            S_on_T = torch.concat(S_on_T_lst, dim=0)
            if use_dynamic_bsz:
                S_on_T = restore_dynamic_batch(S_on_T, batch_idx_list)
        
        # 3. Compute rm_scores on GPU
        # Move all necessary tensors to GPU (they should already be there if passed from fsdp_workers)
        device = get_device_id()
        S_ids = data.batch["student_top_k_ids"].to(device)
        S_logp = data.batch["student_top_k_log_probs"].to(device)
        T_on_S = data.batch["teacher_on_student_log_probs"].to(device)
        
        T_ids = data.batch.get("teacher_top_k_ids", None)
        if T_ids is not None: T_ids = T_ids.to(device)
        T_logp = data.batch.get("teacher_top_k_log_probs", None)
        if T_logp is not None: T_logp = T_logp.to(device)
        overlap_mask = data.batch.get("overlap_mask", None)
        if overlap_mask is not None: overlap_mask = overlap_mask.to(device)

        def compute_reward_weights(S_logp, T_logp, valid_mask, weight_mode, normalize=True):
            """Compute weights for reward calculation.
            
            Args:
                S_logp: Student log probabilities (batch, seq, K)
                T_logp: Teacher log probabilities (batch, seq, K)
                valid_mask: Boolean mask for valid tokens (batch, seq, K)
                weight_mode: "student_p", "teacher_p", or "none"
                normalize: If True, apply softmax normalization across K dim.
                          If False, use raw probabilities (masked by valid_mask).
            
            Returns:
                Weights (batch, seq, K)
            """
            if weight_mode == "student_p":
                log_probs = S_logp
            elif weight_mode == "teacher_p":
                log_probs = T_logp
            elif weight_mode == "none":
                # "none" mode: use a uniform distribution
                log_probs = torch.zeros_like(S_logp)
            else:
                raise ValueError(f"Unknown reward_weight_mode: {weight_mode}")
            
            log_probs = torch.where(valid_mask, log_probs, torch.full_like(log_probs, -float('inf')))
            
            if normalize:
                norm_log_weights = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
                weights = torch.exp(norm_log_weights)
            else:
                weights = torch.exp(log_probs)
            
            weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            
            return weights

        res_tensors = {}
        
        if strategy == "only_stu":
            kl_val = S_logp - T_on_S
            valid_mask = torch.ones_like(S_logp, dtype=torch.bool)
            norm_weights = compute_reward_weights(S_logp, T_on_S, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
            
        elif strategy == "only_tch":
            kl_val = S_on_T - T_logp
            valid_mask = torch.ones_like(S_on_T, dtype=torch.bool)
            norm_weights = compute_reward_weights(S_on_T, T_logp, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
            res_tensors["union_top_k_ids"] = T_ids
            
        elif strategy == "intersection":
            valid_mask = overlap_mask.bool()
            kl_val = S_logp - T_on_S
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp, T_on_S, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
            
        elif strategy == "union":
            union_ids = torch.cat([S_ids, T_ids], dim=-1)
            S_logp_union = torch.cat([S_logp, S_on_T], dim=-1)
            T_logp_union = torch.cat([T_on_S, T_logp], dim=-1)
            
            T_in_S = data.batch["teacher_in_student_mask"].bool().to(device)
            valid_mask = torch.cat([
                torch.ones_like(S_ids, dtype=torch.bool),
                ~T_in_S
            ], dim=-1)
            
            kl_val = S_logp_union - T_logp_union
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp_union, T_logp_union, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
            
            # Use different keys to avoid conflict with batch's student_top_k_ids
            res_tensors["union_top_k_ids"] = union_ids
            res_tensors["union_top_k_log_probs"] = S_logp_union
            res_tensors["student_log_probs_on_teacher_ids"] = S_on_T
        
        elif strategy == "union-intersection":
            union_ids = torch.cat([S_ids, T_ids], dim=-1)
            S_logp_union = torch.cat([S_logp, S_on_T], dim=-1)
            T_logp_union = torch.cat([T_on_S, T_logp], dim=-1)

            S_in_T = overlap_mask.bool().to(device)
            T_in_S = data.batch["teacher_in_student_mask"].bool().to(device)
            valid_mask = torch.cat([
                ~S_in_T,    # S_ids is valid if not in T
                ~T_in_S     # T_ids is valid if not in S
            ], dim=-1)
                
            kl_val = S_logp_union - T_logp_union
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp_union, T_logp_union, valid_mask, reward_weight_mode, normalize=False)
            rm_scores = -kl_val * norm_weights
            
            # Use different keys to avoid conflict with batch's student_top_k_ids
            res_tensors["union_top_k_ids"] = union_ids
            res_tensors["union_top_k_log_probs"] = S_logp_union
            res_tensors["student_log_probs_on_teacher_ids"] = S_on_T
            
        res_tensors["rm_scores"] = rm_scores
        return DataProto.from_dict(tensors=res_tensors)

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    # ------------------------------------------------------------------ E030 credit gate
    def _credit_logp_all(self, data, temperature, micro_batch_size, use_dynamic_bsz, max_token_len):
        """log pi(o_t|s_t) for the whole (local) batch, no grad, batch order preserved."""
        import contextlib
        import transformers.integrations.sdpa_attention as _hf_sdpa
        from torch.nn.attention import SDPBackend, sdpa_kernel
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        d = data.select(batch_keys=select_keys)
        micro_batches = d.split(micro_batch_size)   # fixed micro-batching: equal collective counts on every rank
        # Speedup: run the micro-batches in decreasing token count (the i-th micro-batch has a similar length on every rank, so FSDP waits less at each layer); results are restored to the original order.
        # Fix: with FSDP1 the root unit (embed / lm_head / final norm) does not free the full weights after a no-grad forward, so the next forward skips the all-gather and
        #   the sync_fd / perturb writes into the local shard would not be visible. Reshard(True) before and after every pass, the same thing fsdp_workers does for actor / ref.
        # Speedup 2 (fp32 FD copy): cut the left padding exactly and set the mask to all ones -> HF takes the is_causal=True path and builds no dense mask,
        #   memory-efficient SDPA skips the upper triangle; force repeat_kv (otherwise enable_gqa makes fp32 fall back to the math kernel); round the total width up to 1024.
        #   The right padding sits after the real tokens, so under causal attention it cannot change them; the only difference from the original path is fp32 rounding order.
        root = getattr(self.actor_module, "_handle", None)
        if root is not None and hasattr(root.flat_param, "_local_shard"):   # never forwarded yet (no lazy init): resharding is neither needed nor possible
            root.reshard(True)
        fp32 = bool(getattr(self, "fp32_forward", False))
        ntok = [int(mb.batch["attention_mask"].sum()) for mb in micro_batches]
        out = [None] * len(micro_batches)
        gqa0 = _hf_sdpa.use_gqa_in_sdpa
        if fp32:
            _hf_sdpa.use_gqa_in_sdpa = lambda *a, **k: False
        try:
            for j in sorted(range(len(micro_batches)), key=lambda x: -ntok[x]):
                mb = micro_batches[j].to(get_device_id())
                model_inputs = {**mb.batch, **mb.non_tensor_batch}
                R = model_inputs["responses"].size(-1)
                am = model_inputs["attention_mask"]
                P = am.size(-1) - R
                r = max(int(am[:, P:].sum(-1).max()), 1)
                first = (am[:, :P] > 0).float().argmax(-1)
                s0 = int(first.min())
                if fp32 and int(first.max()) == s0:
                    rr = -(-(P - s0 + r) // 1024) * 1024 - (P - s0)
                    ids, pos = model_inputs["input_ids"][:, s0:], model_inputs["position_ids"][..., s0:]
                    if rr > R:
                        k = rr - R
                        ids = torch.cat([ids, ids[:, -1:].expand(-1, k)], dim=-1)
                        step = torch.arange(1, k + 1, device=pos.device, dtype=pos.dtype)
                        pos = torch.cat([pos, pos[..., -1:] + step], dim=-1)
                    ids, pos = ids[:, :P - s0 + rr], pos[..., :P - s0 + rr]
                    model_inputs["input_ids"], model_inputs["position_ids"] = ids, pos
                    model_inputs["attention_mask"] = torch.ones_like(ids)
                    model_inputs["responses"] = ids[:, P - s0:]
                    ctx = sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]) if ids.is_cuda else contextlib.nullcontext()
                else:
                    rr = min(R, -(-r // 1024) * 1024)
                    s0 = s0 // 256 * 256
                    for key in ("input_ids", "attention_mask", "position_ids"):
                        model_inputs[key] = model_inputs[key][..., s0:P + rr]
                    model_inputs["responses"] = model_inputs["responses"][:, :rr]
                    ctx = contextlib.nullcontext()
                with ctx, torch.no_grad():
                    _, lp, *_ = self._forward_micro_batch(model_inputs, temperature=temperature, calculate_entropy=False)
                out[j] = lp[:, :R] if rr >= R else torch.nn.functional.pad(lp, (0, R - rr))
        finally:
            _hf_sdpa.use_gqa_in_sdpa = gqa0
        if root is not None and hasattr(root.flat_param, "_local_shard"):
            root.reshard(True)
        return torch.concat(out, dim=0)

    def _credit_tangent(self, data, weights, temperature, micro_batch_size, use_dynamic_bsz, max_token_len):
        """Accumulate grad of  -sum_i w_i sum_t log pi(o_{i,t}|s_{i,t})  into the (sharded) .grad buffers."""
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "response_mask"]
        d = data.select(batch_keys=select_keys)
        d.batch["credit_w"] = weights
        # FIXED micro-batching: every rank must run the same number of FSDP forward/backward calls
        # (all-gather / reduce-scatter are collectives). Dynamic bsz would give ranks different counts.
        # Speedup: micro-batches whose weights are all zero (not in this half, or the whole group has A=0) are skipped; the rest run in decreasing token count;
        #   the number of passes is the maximum over ranks, padded with this rank's shortest micro-batch times zero (zero gradient, only to keep the collective count aligned).
        micro_batches = d.split(micro_batch_size)
        ntok = lambda m: int(m.batch["attention_mask"].sum())
        keep = sorted([m for m in micro_batches if bool((m.batch["credit_w"] != 0).any())], key=ntok, reverse=True)
        n = torch.tensor([max(len(keep), 1)], device=get_device_id())
        torch.distributed.all_reduce(n, op=torch.distributed.ReduceOp.MAX)
        dummy = min(micro_batches, key=ntok)
        for i in range(int(n.item())):
            real = i < len(keep)
            mb = (keep[i] if real else dummy).to(get_device_id())
            model_inputs = {**mb.batch, **mb.non_tensor_batch}
            w = model_inputs["credit_w"].float() * (1.0 if real else 0.0)
            _, lp, *_ = self._forward_micro_batch(model_inputs, temperature=temperature, calculate_entropy=False)
            mask = model_inputs["response_mask"].float()
            loss = -(w[:, None] * lp * mask).sum()
            loss.backward()

    @torch.no_grad()
    def _credit_iota_per_prompt(self, data, jvp_mod, gate, mask, w_resp, metrics, need=None):
        """Per-prompt influence. Everything else is as in the batch mode (JVP, the actor AdamW v^L, GRPO-style credit_w, iw_gate).
        For token t of prompt g, iota_t = <u_g, grad log pi(y_t|s_t)> with u_g = P * G_g:
            G_g = sum_{i in g} credit_w_i sum_t grad log pi(o_it)   current-step tangent of this prompt's 8 responses (ascent direction)
            P   = 1/(sqrt(vhat^L)+eps), vhat^L = the actor AdamW exp_avg_sq (bias-corrected); P = 0 wherever v == 0
        In batch mode u is an EMA of the full-batch tangent, where the sampling noise of the other 127 prompts drowns each token's own-prompt signal (measured cos(u_{s-1}, G_s) ~ 0).
        The driver keeps the 8 responses of one prompt on the same rank (CREDIT_SCOPE=prompt), so G_g is formed locally on the bf16 replica and the loop needs no collectives.
        The tangent uses sdpa (flash, forced repeat_kv) with gradient checkpointing for the backward; the influence itself still goes through jvp_influence.influence."""
        import time as _time
        import transformers.integrations.sdpa_attention as _hf_sdpa
        from verl.workers.actor import jvp_influence as _jv
        t0 = _time.time()
        B, R = mask.shape
        opt = getattr(self, "actor_optimizer", None)
        P_sh = []
        for p in gate.view._raw():
            st = opt.state[p]
            v = st["exp_avg_sq"]
            v = v.to_local() if hasattr(v, "to_local") else v
            step = st.get("step", 1)
            step = float(step.item() if torch.is_tensor(step) else step)
            b2 = float(opt.param_groups[0]["betas"][1])
            vh = v.float() / max(1.0 - b2 ** step, 1e-12)
            P_sh.append(torch.where(vh > 0, 1.0 / (vh.sqrt() + gate.adam_eps), torch.zeros_like(vh)))
        jvp_mod.sync_weights(self.actor_module)
        P = jvp_mod.gather_tangent(gate.view, P_sh)
        n_full = sum(float((t.float() ** 2).sum()) for t in P.values()) ** 0.5
        n_sh = gate.norm(P_sh)
        del P_sh
        jvp_mod.u_full = None
        metrics["credit/u_gather_rel_err"] = abs(n_full - n_sh) / max(n_sh, 1e-30)
        assert metrics["credit/u_gather_rel_err"] < 0.05, "gathered P norm %.4g vs sharded %.4g" % (n_full, n_sh)
        t1 = _time.time()
        rep = jvp_mod.replica
        if not getattr(rep, "_e031pp_gc", False):
            rep.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            rep._e031pp_gc = True
        gid = data.batch["credit_gid"].long()
        ids_all, am_all = data.batch["input_ids"], data.batch["attention_mask"]
        pos_all, resp_all = data.batch["position_ids"], data.batch["responses"]
        Q = int(_jv._QBLOCK)
        D = torch.zeros(B, R, device=mask.device, dtype=torch.float32)
        n_groups = 0
        n_rows = 0
        t_bwd = 0.0
        t_jvp = 0.0
        for g in torch.unique(gid).tolist():
            rows = torch.nonzero(gid == g, as_tuple=True)[0]
            if not bool((w_resp[rows] != 0).any()):
                continue
            n_groups += 1
            ta = _time.time()
            impl0, gqa0 = rep.config._attn_implementation, _hf_sdpa.use_gqa_in_sdpa
            for p in rep.parameters():
                p.requires_grad_(True)
                p.grad = None
            rep.config._attn_implementation = "sdpa"
            _hf_sdpa.use_gqa_in_sdpa = lambda *a, **k: False
            rep.train()
            try:
                with torch.enable_grad():
                    for b in rows.tolist():
                        wb = float(w_resp[b])
                        if wb == 0.0:
                            continue
                        am = am_all[b]
                        valid = am.nonzero(as_tuple=True)[0]
                        n_resp = int(am[-R:].sum())
                        if valid.numel() == 0 or n_resp == 0:
                            continue
                        lo, hi = int(valid[0]), int(valid[-1]) + 1
                        ids = ids_all[b:b + 1, lo:hi]
                        pos = pos_all[b:b + 1, lo:hi] if pos_all.dim() == 2 else pos_all[:, b:b + 1, lo:hi]
                        y = resp_all[b, :n_resp]
                        s = (hi - lo) - n_resp - 1
                        h = rep.model(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=pos,
                                      use_cache=False).last_hidden_state[0, s:s + n_resp]
                        hd = h.detach().requires_grad_(True)
                        W = rep.lm_head.weight
                        for i in range(0, n_resp, Q):
                            lp = torch.log_softmax((hd[i:i + Q] @ W.t()).float(), -1).gather(-1, y[i:i + Q, None])[:, 0]
                            (lp.sum() * wb).backward()
                        h.backward(hd.grad)
                        n_rows += 1
                        del h, hd
            finally:
                rep.config._attn_implementation = impl0
                _hf_sdpa.use_gqa_in_sdpa = gqa0
                rep.eval()
            u_g = {}
            for n, p in rep.named_parameters():
                gr = p.grad if p.grad is not None else torch.zeros_like(p)
                u_g[n] = gr.mul_(P[n])
                p.grad = None
                p.requires_grad_(False)
            tb = _time.time()
            t_bwd += tb - ta
            D[rows] = jvp_mod.influence(ids_all[rows], am_all[rows],
                                        pos_all[rows] if pos_all.dim() == 2 else pos_all[:, rows],
                                        resp_all[rows], u_full=u_g, need=None if need is None else need[rows])
            del u_g
            t_jvp += _time.time() - tb
        del P
        torch.cuda.empty_cache()
        st = _all_reduce_sum(torch.tensor([float(n_groups), float(n_rows)], device=mask.device, dtype=torch.float64))
        metrics["credit/pp_mixed_groups"] = float(st[0])
        metrics["credit/pp_rows_bwd"] = float(st[1])
        metrics["credit/jvp_sync_s"] = t1 - t0
        metrics["credit/pp_bwd_s"] = t_bwd
        metrics["credit/jvp_pass_s"] = t_jvp
        return D

    def _compute_credit_prompt(self, data, gate, mask, ell, w_resp, A, metrics):
        """Per-prompt credit. No full-batch tangent and no shadow moments (unused in this mode), so the credit cost matches the batch path.
        c_t = ell_t * iota_t with iota the per-prompt JVP influence; iw_gate uses z = c (CREDIT_Z=c, the default; z=iota reproduces the original code path):
        the first-order gain of the weighted OPD update along u is sum w c = FO + (lam/sigma) sum c^2 >= FO.
        The active set keeps only tokens with iota != 0 (mixed-reward groups); otherwise the zeros of non-mixed groups shrink sigma and inflate lam."""
        opt = getattr(self, "actor_optimizer", None)
        ready = opt is not None and all(("exp_avg_sq" in opt.state.get(p, {})) for p in gate.view._raw())
        not_ready = _all_reduce_sum(torch.tensor([0.0 if ready else 1.0], device=mask.device, dtype=torch.float64))
        if float(not_ready) > 0:
            D = torch.zeros_like(ell)
            c = torch.zeros_like(ell)
            wmask = mask.clone()
            metrics["credit/active"] = 0.0
        else:
            from verl.workers.actor.credit_gate import iw_gate
            self.actor_module.eval()
            p_y_all = data.batch["old_log_probs"].float().exp()
            mixr = (w_resp != 0).float()[:, None] * mask
            _v = mixr.bool() & (ell != 0)
            _thr = gate.global_quantile(p_y_all, _v, float(data.meta_info.get("credit_active_frac", 0.2) or 0.2))
            need = (_v & (p_y_all <= _thr)) if os.environ.get("CREDIT_JVP_NEED", "1") == "1" else None
            D = self._credit_iota_per_prompt(data, self.jvp_influence, gate, mask, w_resp, metrics, need=need) * mask
            c = ell * D * mask
            _len = mask.sum(-1).long().clamp_min(1)
            last_tok = torch.zeros_like(mask, dtype=torch.bool)
            last_tok[torch.arange(mask.size(0), device=mask.device), _len - 1] = True
            last_tok &= mask.bool()
            use_c = os.environ.get("CREDIT_Z", "c") == "c"
            mi = data.meta_info
            wmask, gstats = iw_gate(c if use_c else D, mask, ell * mixr, p_y_all,
                                    active_frac=float(mi.get("credit_active_frac", 0.2) or 0.2),
                                    w_max=float(mi.get("credit_w_max", 1.5) or 1.5),
                                    f_hi=float(mi.get("credit_f_hi", 0.05) or 0.05),
                                    f_lo=float(mi.get("credit_f_lo", 0.15) or 0.15),
                                    lam_cap=float(mi.get("credit_lam_cap", 4.0) or 4.0),
                                    last_tok=last_tok,
                                    lam_fixed=float(mi.get("credit_lam_fixed", 0.0) or 0.0),
                                    w_min=float(mi.get("credit_w_min", 0.0) or 0.0))
            metrics.update(gstats)
            num = _all_reduce_sum((wmask * c).sum().double())
            den = _all_reduce_sum(c.sum().double())
            aden = _all_reduce_sum(c.abs().sum().double())
            n_tok = _all_reduce_sum(mask.sum().double()).clamp_min(1)
            metrics["credit/gain_ratio_vs_plain"] = float(num / den) if abs(float(den)) > 1e-12 else float("nan")
            metrics["credit/gain_delta_rel"] = float((num - den) / aden.clamp_min(1e-30))
            metrics["credit/rms_iota"] = float(torch.sqrt(_all_reduce_sum((D * D * mask).sum().double()) / n_tok))
            metrics["credit/frac_w0"] = float(_all_reduce_sum(((1 - wmask) * mask).sum().double()) / n_tok)
            metrics["credit/sum_c_over_abs_all"] = gate.signed_share(c, mask)
            metrics["credit/z_uses_c"] = 1.0 if use_c else 0.0
            metrics["credit/active"] = 1.0
        self.actor_module.train()
        return DataProto.from_dict(tensors={"credit_mask": wmask, "credit": c, "credit_D": D},
                                   meta_info={"metrics": metrics})

    def compute_credit(self, data: DataProto) -> DataProto:
        """E030: per-token credit mask (see credit_gate.py). Expects in data.batch:
        rm_scores (ell_t), old_log_probs, response_mask, credit_w (A_i/m_i, per response),
        credit_half (0/1 per response), credit_A (A_i). Returns credit_mask / credit / credit_D
        (B, T) and global metrics in meta_info["metrics"]."""
        gate = self.credit_gate
        temperature = data.meta_info["temperature"]
        micro_batch_size = data.meta_info["micro_batch_size"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        max_token_len = data.meta_info.get("max_token_len", 0) * self.ulysses_sequence_parallel_size
        torch.cuda.empty_cache()   # free the fragmented cache: close to the memory ceiling the allocator otherwise keeps freeing and re-requesting (16% of the profile)
        dev = get_device_id()
        data = data.to(dev)
        mask = data.batch["response_mask"].float()
        ell = data.batch["rm_scores"].float()
        w_resp = data.batch["credit_w"].float()
        half = data.batch["credit_half"].long()
        A = data.batch["credit_A"].float()
        metrics = {}
        if os.environ.get("CREDIT_SCOPE", "batch") == "prompt" and getattr(self, "jvp_influence", None) is not None:
            return self._compute_credit_prompt(data, gate, mask, ell, w_resp, A, metrics)

        # ---- 1. Dr.GRPO tangent, in two random halves (same FLOPs as one backward)
        self.actor_module.train()
        gate.zero_grads()
        halves = []
        for h in ((0, 1) if gate.split_half else (None,)):
            wh = w_resp if h is None else w_resp * (half == h).float()
            self._credit_tangent(data, wh, temperature, micro_batch_size, use_dynamic_bsz, max_token_len)
            halves.append(gate.snapshot_grads())
            gate.zero_grads()
        G = halves[0] if len(halves) == 1 else [a + b for a, b in zip(halves[0], halves[1])]
        if len(halves) == 2:
            metrics["credit/cos_split_half"] = gate.cos(halves[0], halves[1])
            metrics["credit/G_norm_half0"] = gate.norm(halves[0])
        metrics["credit/G_norm"] = gate.norm(G)
        del halves
        torch.cuda.empty_cache()   # free the fragmented cache before the FD pass allocates a new set of large tensors

        # ---- 2. direction from the PREVIOUS steps' moments, then the finite difference
        self.actor_module.eval()
        gate_kind = str(data.meta_info.get("credit_gate_kind", "iw") or "iw")
        u = gate.direction_from_optimizer(getattr(self, "actor_optimizer", None))   # E031: u = m_hat/(sqrt(v_hat^L)+eps)
        if u is None:
            D = torch.zeros_like(ell)
            c = torch.zeros_like(ell)
            wmask = mask.clone()                      # step 0: plain OPD
            metrics["credit/active"] = 0.0
        else:
            metrics["credit/cos_u_G"] = gate.cos(u, G)
            metrics["credit/u_norm"] = gate.norm(u)
            jvp_mod = getattr(self, "jvp_influence", None)
            if jvp_mod is not None:
                # ---- E030 JVP path: one forward-mode pass on the bf16 non-FSDP replica (jvp_influence.py)
                import time as _time
                _t = _time.time()
                if os.environ.get("CREDIT_SCOPE", "batch") == "prompt":   # per-prompt influence
                    del u
                    D = self._credit_iota_per_prompt(data, jvp_mod, gate, mask, w_resp, metrics) * mask
                    dlp = D * (2.0 * gate.eps)
                else:
                    jvp_mod.sync_weights(self.actor_module)
                    jvp_mod.gather_tangent(gate.view, u)
                    # sanity: the gathered full u must have the norm of the sharded u (catches a wrong FSDP gather)
                    _n_full = sum(float((t.float() ** 2).sum()) for t in jvp_mod.u_full.values()) ** 0.5
                    _n_shard = gate.norm(u)
                    metrics["credit/u_gather_rel_err"] = abs(_n_full - _n_shard) / max(_n_shard, 1e-30)
                    assert metrics["credit/u_gather_rel_err"] < 0.05, "gathered u norm %.4g vs sharded %.4g" % (_n_full, _n_shard)
                    metrics["credit/jvp_sync_s"] = _time.time() - _t
                    del u
                    _t = _time.time()
                    _need = None
                    if os.environ.get("CREDIT_JVP_NEED", "1") == "1" and gate_kind == "iw":   # speedup: only compute the active set
                        _p = data.batch["old_log_probs"].float().exp()
                        _v = mask.bool() & (ell != 0)
                        _thr = gate.global_quantile(_p, _v, float(data.meta_info.get("credit_active_frac", 0.2) or 0.2))
                        _need = _v & (_p <= _thr)
                    D = jvp_mod.influence(data.batch["input_ids"], data.batch["attention_mask"],
                                          data.batch["position_ids"], data.batch["responses"], need=_need) * mask
                    metrics["credit/jvp_pass_s"] = _time.time() - _t
                    jvp_mod.u_full = None
                    torch.cuda.empty_cache()
                    dlp = D * (2.0 * gate.eps)     # same telemetry scale as the FD path (rms_dlp = 2*eps*rms(D))
                # ---- optional cross-check against the fp32 finite difference on the first N active steps
                n_chk = int(data.meta_info.get("credit_jvp_check", 0))
                fd_chk = getattr(self, "fd_policy", None)
                if fd_chk is not None and gate.n_upd <= n_chk and os.environ.get("CREDIT_SCOPE", "batch") != "prompt":
                    u2 = gate.direction_from_optimizer(getattr(self, "actor_optimizer", None))   # the check must use the same direction that produced D
                    _s2 = (float(gate.numel) ** 0.5) / max(gate.norm(u2), 1e-30)
                    u2 = [x * _s2 for x in u2]
                    tf32_m, tf32_c = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
                    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
                    try:
                        gate.sync_fd(); fd_chk.actor_module.eval()
                        fd_mbs = int(data.meta_info.get("credit_fd_micro_batch_size", micro_batch_size))
                        gate.perturb(u2, +gate.eps)
                        lp_p = fd_chk._credit_logp_all(data, temperature, fd_mbs, False, max_token_len)
                        gate.perturb(u2, -2.0 * gate.eps)
                        lp_m = fd_chk._credit_logp_all(data, temperature, fd_mbs, False, max_token_len)
                        gate.perturb(u2, +gate.eps)
                    finally:
                        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32_m, tf32_c
                    del u2
                    D_fd = (lp_p.float() - lp_m.float()) / (2.0 * gate.eps * _s2) * mask
                    both = mask * (D != 0).float() * (D_fd != 0).float()
                    agree = ((torch.sign(D) == torch.sign(D_fd)).float() * both).sum().double()
                    metrics["credit/jvp_fd_sign_agree"] = float(_all_reduce_sum(agree) / _all_reduce_sum(both.sum().double()).clamp_min(1))
                    _ckm = _need.float() if _need is not None else mask; num = _all_reduce_sum(((D - D_fd) ** 2 * _ckm).sum().double()); den = _all_reduce_sum((D_fd ** 2 * _ckm).sum().double())
                    metrics["credit/jvp_fd_rel_rmse"] = float(torch.sqrt(num / den.clamp_min(1e-30)))
                    del D_fd, lp_p, lp_m
            else:
                fd = getattr(self, "fd_policy", None)          # fp32 copy (calibration 09-10); else the actor itself
                fd_actor = fd if fd is not None else self
                tf32_m, tf32_c = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
                torch.backends.cuda.matmul.allow_tf32 = False   # TF32 rounds a 1e-6 perturbation away (0.886 agreement)
                torch.backends.cudnn.allow_tf32 = False
                try:
                    gate.sync_fd()
                    fd_actor.actor_module.eval()
                    fd_mbs = int(data.meta_info.get("credit_fd_micro_batch_size", micro_batch_size))   # fp32 logits are big
                    gate.perturb(u, +gate.eps)
                    lp_plus = fd_actor._credit_logp_all(data, temperature, fd_mbs, False, max_token_len)
                    gate.perturb(u, -2.0 * gate.eps)
                    lp_minus = fd_actor._credit_logp_all(data, temperature, fd_mbs, False, max_token_len)
                    gate.perturb(u, +gate.eps)
                finally:
                    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32_m, tf32_c
                del u
                D = (lp_plus.float() - lp_minus.float()) / (2.0 * gate.eps) * mask
                dlp = (lp_plus.float() - lp_minus.float()) * mask
            c = ell * D * mask
            p_y_all = data.batch["old_log_probs"].float().exp()
            if gate_kind == "iw":
                # ---- E031: influence-weighted gate on the active set (credit_gate.iw_gate)
                from verl.workers.actor.credit_gate import iw_gate
                _len = mask.sum(-1).long().clamp_min(1)
                last_tok = torch.zeros_like(mask, dtype=torch.bool)
                last_tok[torch.arange(mask.size(0), device=mask.device), _len - 1] = True
                last_tok &= mask.bool()
                _use_c = os.environ.get("CREDIT_Z", "c") == "c"   # z = c = ell * iota
                wmask, gstats = iw_gate(c if _use_c else D, mask, ell, p_y_all,
                                        active_frac=float(data.meta_info.get("credit_active_frac", 0.2) or 0.2),
                                        w_max=float(data.meta_info.get("credit_w_max", 1.5) or 1.5),
                                        f_hi=float(data.meta_info.get("credit_f_hi", 0.05) or 0.05),
                                        f_lo=float(data.meta_info.get("credit_f_lo", 0.15) or 0.15),
                                        lam_cap=float(data.meta_info.get("credit_lam_cap", 4.0) or 4.0),
                                        last_tok=last_tok,
                                        lam_fixed=float(data.meta_info.get("credit_lam_fixed", 0.0) or 0.0),
                                        w_min=float(data.meta_info.get("credit_w_min", 0.0) or 0.0))
                metrics.update(gstats)
                # realised first-order gain relative to plain OPD:  sum_t w_t d_t iota_t / sum_t d_t iota_t
                _num = _all_reduce_sum((wmask * c).sum().double()); _den = _all_reduce_sum(c.sum().double())
                metrics["credit/gain_ratio_vs_plain"] = float(_num / _den) if abs(float(_den)) > 1e-12 else float("nan")
                metrics["credit/rms_iota"] = float(torch.sqrt(_all_reduce_sum((D * D * mask).sum().double()) / _all_reduce_sum(mask.sum().double()).clamp_min(1)))
            else:
                wmask = (c > 0).float() * mask      # E030 hard sign gate
            metrics["credit/active"] = 1.0
            n_tok = _all_reduce_sum((mask.sum()).double())
            metrics["credit/rms_dlp"] = float(torch.sqrt(_all_reduce_sum((dlp * dlp).sum().double()) / n_tok.clamp_min(1)))
            metrics["credit/frac_w0"] = float(_all_reduce_sum(((1 - wmask) * mask).sum().double()) / n_tok.clamp_min(1))
            # GRPD (diagonal) mask: sign(A_i) == sign(ell_t); agreement with ours on tokens where A != 0
            grpd = ((torch.sign(A)[:, None] * torch.sign(ell)) > 0).float()
            valid_A = (A != 0).float()[:, None] * mask
            agree = ((grpd == wmask).float() * valid_A).sum().double()
            metrics["credit/grpd_agree"] = float(_all_reduce_sum(agree) / _all_reduce_sum(valid_A.sum().double()).clamp_min(1))
            metrics["credit/sum_c_over_abs_all"] = gate.signed_share(c, mask)
            # lowest-p tokens: p(o_t) = exp(old_log_probs); thresholds are global quantiles
            p_y = p_y_all
            for f in gate.lowp_fracs:
                thr = gate.global_quantile(p_y, mask, f)
                low = ((p_y <= thr).float() * mask)
                metrics["credit/sum_c_over_abs_lowp%d" % int(round(100 * f))] = gate.signed_share(c, low)
                metrics["credit/frac_w0_lowp%d" % int(round(100 * f))] = float(
                    _all_reduce_sum(((1 - wmask) * low).sum().double()) / _all_reduce_sum(low.sum().double()).clamp_min(1))
                metrics["credit/p_thr_lowp%d" % int(round(100 * f))] = thr
            metrics["credit/ell_mean"] = float(_all_reduce_sum((ell * mask).sum().double()) / n_tok.clamp_min(1))
            metrics["credit/ell_neg_frac"] = float(_all_reduce_sum(((ell < 0).float() * mask).sum().double()) / n_tok.clamp_min(1))

        # ---- 3. advance the moments with this step's tangent (post-credit => pre-update credit)
        gate.update_moments(G)
        del G
        metrics["credit/n_upd"] = float(gate.n_upd)
        metrics["credit/eps"] = gate.eps
        self.actor_module.train()
        out = DataProto.from_dict(tensors={"credit_mask": wmask, "credit": c, "credit_D": D},
                                  meta_info={"metrics": metrics})
        return out

    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        top_k = data.meta_info.get("top_k", 0)
        print(f"In compute_log_prob, top_k: {top_k}")
        log_probs_lst = []
        entropy_lst = []
        topk_ids_lst = []
        topk_log_probs_lst = []

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, topk_ids, topk_log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy, top_k=top_k
                )
            # Keep on GPU to avoid expensive CPU-GPU transfer for large top-k
            # log_probs = log_probs.to("cpu")
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                # entropy = entropy.to("cpu")
                entropy_lst.append(entropy)
            if top_k > 0:
                # topk_ids = topk_ids.to("cpu")
                # topk_log_probs = topk_log_probs.to("cpu")
                topk_ids_lst.append(topk_ids)
                topk_log_probs_lst.append(topk_log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        
        topk_ids_tensor = None
        topk_log_probs_tensor = None
        if top_k > 0:
            topk_ids_tensor = torch.concat(topk_ids_lst, dim=0)
            topk_log_probs_tensor = torch.concat(topk_log_probs_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if top_k > 0:
                topk_ids_tensor = restore_dynamic_batch(topk_ids_tensor, batch_idx_list)
                topk_log_probs_tensor = restore_dynamic_batch(topk_log_probs_tensor, batch_idx_list)

        return log_probs, entropys, topk_ids_tensor, topk_log_probs_tensor

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")

        if "format_mask" in data.batch.keys():
            select_keys.append("format_mask") # (bsz, 1)
        
        # Include student_top_k_log_probs if present (for top-k distillation)
        if "student_top_k_log_probs" in data.batch.keys():
            select_keys.append("student_top_k_log_probs")

        # Include student_top_k_ids if present (for fixing "apples-to-oranges" bug)
        if "student_top_k_ids" in data.batch.keys():
            select_keys.append("student_top_k_ids")

        # Include union_top_k_ids/log_probs for union strategy
        if "union_top_k_ids" in data.batch.keys():
            print("Now we are using union strategy, get union_top_k_ids")
            select_keys.append("union_top_k_ids")
            # now we don't need to store student_top_k_ids and student_top_k_log_probs for union strategy
            if "student_top_k_ids" in select_keys:
                select_keys.remove("student_top_k_ids")

        if "union_top_k_log_probs" in data.batch.keys():
            print("Now we are using union strategy, get union_top_k_log_probs")
            select_keys.append("union_top_k_log_probs")
            # now we don't need to store student_top_k_log_probs for union strategy
            if "student_top_k_log_probs" in select_keys:
                select_keys.remove("student_top_k_log_probs")   

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    
                    # Check if we have 3D advantages (top-k sampling case)
                    # If so, we need to recompute top-k log probs for correct gradient
                    if advantages.dim() == 3:
                        top_k = advantages.shape[-1]
                        # For union strategy, use union_top_k_ids; otherwise use student_top_k_ids
                        student_top_k_ids = None
                        if "union_top_k_ids" in model_inputs:
                            student_top_k_ids = model_inputs["union_top_k_ids"]
                        elif "student_top_k_ids" in model_inputs:
                            student_top_k_ids = model_inputs["student_top_k_ids"]

                        entropy, _, _, topk_log_probs = self._forward_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy,
                            top_k=top_k, student_top_k_ids=student_top_k_ids
                        )
                        log_prob_for_loss = topk_log_probs
                        
                    else:
                        _, log_prob, *_ = self._forward_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                        )
                        log_prob_for_loss = log_prob

                    format_mask = None
                    if "format_mask" in model_inputs.keys():
                        format_mask = model_inputs["format_mask"]
            

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            print("on_policy")
                            # For on-policy (ppo_epochs=1), use current policy as "old"
                            # log_prob_for_loss is already 3D for top-k case
                            old_log_prob = log_prob_for_loss.detach()
                        else:
                            print("off_policy")
                            # For off-policy, use stored log probs
                            # For 3D top-k case, use stored log probs (union or student)
                            if advantages.dim() == 3:
                                if "union_top_k_log_probs" in model_inputs:
                                    old_log_prob = model_inputs["union_top_k_log_probs"]
                                elif "student_top_k_log_probs" in model_inputs:
                                    old_log_prob = model_inputs["student_top_k_log_probs"]
                                else:
                                    old_log_prob = model_inputs["old_log_probs"]
                            else:
                                old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # NOTE: Both mismatch diagnostic metrics (PPL, KL, etc.) and IS weight metrics
                    # are computed centrally in ray_trainer.py for consistency and efficiency.
                    # This ensures metrics are computed uniformly across all batches at the trainer level
                    # and avoids redundant computation across workers and micro-batches.

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob_for_loss,  # 3D for top-k, 2D otherwise
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                        format_mask=format_mask,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics

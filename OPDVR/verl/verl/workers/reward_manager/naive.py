# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import multiprocessing
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait as futures_wait
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


# ---------------------------------------------------------------------------
# Code scoring actually runs the generated programs (prime_code spawns a subprocess per response),
#   while the reward manager is called serially inside the driver -- 1024 responses per step would fork 1024 times from a 20 GB driver,
#   which is what used to stall the GPUs. So code items are collected first and judged in parallel by a persistent process pool.
# Math scoring does not take this path and behaves exactly as before.
_CODE_SOURCES = ("codecontests", "apps", "codeforces", "taco")
_code_pool = None


def _code_judge_one(task):
    data_source, solution_str, ground_truth = task
    # The subprocess uses verl's own default_compute_score: the same file and function the launcher passes as custom_reward_function,
    #   because pickling self.compute_score (a function object loaded dynamically from a path) to a subprocess is not reliable.
    from verl.utils.reward_score import default_compute_score

    try:
        return float(default_compute_score(data_source=data_source, solution_str=solution_str, ground_truth=ground_truth))
    except Exception as e:  # a crash inside scoring counts as 0, but print it
        print(f"[code judge] scoring failed ({data_source}): {type(e).__name__}: {e}")
        return 0.0


def _get_code_pool():
    global _code_pool
    if _code_pool is None:
        n = int(os.environ.get("CODE_JUDGE_WORKERS", "0")) or max(4, min(32, (os.cpu_count() or 8) // 4))
        kw = dict(max_workers=n, mp_context=multiprocessing.get_context("fork"))
        try:  # max_tasks_per_child needs py3.11; it recycles workers so judged code cannot pollute them
            _code_pool = ProcessPoolExecutor(max_tasks_per_child=int(os.environ.get("CODE_JUDGE_MAX_TASKS", "64")), **kw)
        except (TypeError, ValueError):  # py<3.11 raises TypeError; the fork context rejects max_tasks_per_child with ValueError
            _code_pool = ProcessPoolExecutor(**kw)
        print(f"[code judge] process pool with {n} workers")
    return _code_pool


def _drop_code_pool():
    global _code_pool
    if _code_pool is not None:
        try:
            _code_pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        _code_pool = None


def judge_code_batch(tasks):
    """Judge a batch of code responses in parallel; returns one score per task, 0 for the ones that time out, and rebuilds the pool."""
    if not tasks:
        return []
    total = float(os.environ.get("CODE_JUDGE_TOTAL_TIMEOUT", "900"))
    scores = [0.0] * len(tasks)
    try:
        pool = _get_code_pool()
        futs = {pool.submit(_code_judge_one, t): i for i, t in enumerate(tasks)}
    except Exception as e:
        print(f"[code judge] pool failed to start, falling back to serial: {type(e).__name__}: {e}")
        return [_code_judge_one(t) for t in tasks]
    done, not_done = futures_wait(list(futs), timeout=total)
    for f in done:
        i = futs[f]
        try:
            scores[i] = float(f.result())
        except Exception as e:
            print(f"[code judge] could not fetch result {i}: {type(e).__name__}: {e}")
    if not_done:
        print(f"[code judge] {len(not_done)}/{len(tasks)} responses did not finish within {total:.0f}s, scored 0, rebuilding the pool")
        _drop_code_pool()
    return scores


@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", enable_format_reward=False) -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
            enable_format_reward: Whether to enable format reward. Defaults to False.
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.enable_format_reward = enable_format_reward

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        # if "rm_scores" in data.batch.keys():
        #     if return_dict:
        #         reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
        #         reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
        #         return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
        #     else:
        #         return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        format_tensor = torch.zeros(data.batch["responses"].shape[0], dtype=torch.float32) # (batch_size, )
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        # judge the code items in parallel first (see the note at the top of this file); results go into code_scores by index
        code_scores = {}
        code_tasks, code_idx = [], []
        for i in range(len(data)):
            item = data[i]
            ds = item.non_tensor_batch[self.reward_fn_key]
            if ds in _CODE_SOURCES:
                pl = item.batch["prompts"].shape[-1]
                vrl = int(item.batch["attention_mask"][pl:].sum())
                rs = self.tokenizer.decode(item.batch["responses"][:vrl], skip_special_tokens=True)
                code_tasks.append((ds, rs, item.non_tensor_batch["reward_model"]["ground_truth"]))
                code_idx.append(i)
        if code_tasks:
            t0 = time.time()
            got = judge_code_batch(code_tasks)
            code_scores = dict(zip(code_idx, got))
            print("[code judge] %d responses, %.0fs, mean %.3f, %d perfect" % (
                len(code_tasks), time.time() - t0, sum(got) / len(got), sum(1 for x in got if x >= 1.0)))

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            if self.enable_format_reward:
                if r"\boxed" in response_str:
                    format_score = 1.0
                else:
                    format_score = 0.0
                format_tensor[i] = format_score

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            if i in code_scores:
                score = code_scores[i]
            else:
                score = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        # Caculate the reward using reward_fn first, we want to know true reward scores, but we still use rm_scores for training
        if "rm_scores" in data.batch.keys():
            print(f"Now we are using rm_scores!")
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                reward_extra_info["true_reward_score"] = reward_tensor
                if self.enable_format_reward:
                    print("Format mask has been added to reward_extra_info!")
                    reward_extra_info["format_mask"] = format_tensor
                print("True reward score has been added to reward_extra_info!")
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

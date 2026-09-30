#!/bin/bash
# Shared settings for the math distillation methods.
# Source this, then call one of opd_baseline.sh / grpd.sh / opdvr.sh with the arm's switches.
# Everything here is an env override consumed by the training entry scripts (they all use ${VAR:-default}).
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-Qwen/Qwen3-4B}                                  # nothink via enable_thinking=False in the scripts
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500}   # teacher (Step500; != Step1200)
export TRAIN_DATASET=${TRAIN_DATASET:-$(pwd)/datasets/deepmath-level6-train.parquet}       # == Keven16/G-OPD-Training-Data level6 (57,046 rows)
export TEST_DATASET=${TEST_DATASET:-$(pwd)/datasets/valid_final_unique.parquet}            # six evaluation sets (1,590 prompts)
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-128}     # prompts per step (= train_batch_size with PARALLEL_SIZE=1)
export N_RESPONSES=${N_RESPONSES:-8}               # rollouts per prompt -> 1024 rollouts/step
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-12288}      # 12K response tokens
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-12288}
export ACTOR_LR=${ACTOR_LR:-1e-5}                  # full fine-tuning learning rate
export MODEL_DTYPE=${MODEL_DTYPE:-float32}         # fp32 master weights, bf16 compute
export LOG_PROB_TOP_K=0                            # sampled-token OPD for every arm
export GRPO_NORM_BY_STD=${GRPO_NORM_BY_STD:-False} # Dr.GRPO (irrelevant to the sign masks; affects GRPD's |A| scale)
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-14336}
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-14336}
export TEST_FREQ=${TEST_FREQ:-10}
export SAVE_FREQ=${SAVE_FREQ:-50}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export PROJECT_NAME=${PROJECT_NAME:-Dropd}
# extra hydra overrides appended to the entry script via "$@"
ARM_EXTRA=(
  actor_rollout_ref.actor.grad_clip=${GRAD_CLIP:-1.0}   # gradient-clipping threshold; 1.0 is the framework default
  trainer.total_training_steps=${STEPS:-100}
  data.shuffle=True data.seed=${DATA_SEED:-1}                      # reproducible data shuffling
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOGP_MBS:-1}   # logits [mbs x 13K x 152K] bf16 = 4 GB each; G-OPD uses 4
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOGP_MBS:-1}
  reward_model.micro_batch_size_per_gpu=${LOGP_MBS:-1}
  +actor_rollout_ref.rollout.credit_fd_micro_batch_size_per_gpu=1  # fp32 copy: 8 GB of logits per sequence
  trainer.val_before_train=${VAL_BEFORE_TRAIN:-False}   # no step-0 evaluation
  'trainer.logger=["console","wandb"]'
  actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_UTIL:-0.7}
  actor_rollout_ref.rollout.enforce_eager=False
  actor_rollout_ref.rollout.free_cache_engine=True
  data.val_batch_size=${VAL_BATCH:-400}   # evaluation runs in batches and prints one line per batch (progress); null = a single batch
  actor_rollout_ref.model.enable_activation_offload=${ACT_OFFLOAD:-False}   # trade GPU memory for activation-transfer overhead
  ++actor_rollout_ref.rollout.val_kwargs.max_tokens=${MAX_VAL_RESP_LENGTH}   # if unset this is None, and each worker pads to its own longest response so the concatenation mismatches
  actor_rollout_ref.rollout.val_kwargs.n=${VAL_N:-4}   # avg@N at evaluation time
  actor_rollout_ref.rollout.dtype=${ROLLOUT_DTYPE:-bfloat16}   # MODEL_DTYPE=float32 only concerns the actor master weights; generation stays bf16
  ++reward_model.model.dtype=${TEACHER_DTYPE:-bfloat16}   # the teacher forward is autocast bf16 anyway: same numbers, half the offload traffic
)
# >>> fewshot extra: use a fixed few-shot text verbatim as the prompt, with the matching stop rules (off by default)
if [ "${FEWSHOT:-0}" = 1 ]; then
  ARM_EXTRA+=( +data.raw_prompt_text=True +actor_rollout_ref.rollout.fewshot_stop=True )
fi
# <<< fewshot extra

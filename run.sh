#!/bin/bash
# Launcher for the math distillation experiments (single node, 8 GPUs).
#
#   bash run.sh setup                     # venv + models (~30 min)
#   bash run.sh data                      # verify the supplied math parquet files
#   bash run.sh train dropd               # Dr. OPD with the reported defaults (lambda 0.4, seed 2)
#   bash run.sh train dropd 0.4 1         # <arm> [lambda, dropd only] [seed]
#   bash run.sh train plain               # baseline: sampled-token on-policy distillation
#   bash run.sh train grpd                # baseline: correctness-gated distillation
#   bash run.sh train opdgrpo             # baseline: distillation reward + GRPO advantage
#   bash run.sh train exopd               # baseline: extrapolated distillation reward
#
# Defaults: student Qwen3-1.7B (non-thinking),
# teacher Qwen3-4B (non-thinking), AdamW lr 1e-5 constant without warmup, grad_clip 1.0,
# 128 prompts x 8 rollouts per step, prompt limit 1024, response limit 12288, bf16 rollout,
# fp32 master weights, Dr.GRPO advantage, 50 steps, checkpoints at steps 30/40/50.
# The original trainer ignores TEST_STEPS; see docs/reproduction.md before comparing scores.
# Any of them can be overridden on the command line, e.g.
#   STEPS=30 EVAL_STEPS=10,20,30 bash run.sh train dropd 0.4 2
set -euo pipefail
DRY_RUN=0
POSITIONAL=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --help|-h) printf 'Usage: bash run.sh setup|data|train <dropd|plain|grpd|opdgrpo|exopd> [lambda] [data_seed] [--dry-run]\n'; exit 0 ;;
    --*) printf 'Unknown option: %s\n' "$arg" >&2; exit 2 ;;
    *) POSITIONAL+=("$arg") ;;
  esac
done
set -- "${POSITIONAL[@]}"
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-$HERE/work}                      # venv, model and HF caches, checkpoints
VENV=${VENV:-$WORK/venv}
RESULTS_DIR=${RESULTS_DIR:-$HERE/results}
export RESULTS_DIR

STUDENT=${STUDENT:-Qwen/Qwen3-1.7B}           # student; a local path works too
TEACHER=${TEACHER:-Qwen/Qwen3-4B}             # teacher, scores the student's own tokens
STEPS=${STEPS:-50}
EVAL_STEPS=${EVAL_STEPS:-30,40,50}            # checkpoint steps; does not control evaluation timing
RESP_LEN=${RESP_LEN:-12288}
VAL_LEN=${VAL_LEN:-12288}
GRAD_CLIP=${GRAD_CLIP:-1.0}
CREDIT_ACTIVE=${CREDIT_ACTIVE:-1.0}           # Dr. OPD: active set = every token of a mixed-reward group
CREDIT_WMIN=${CREDIT_WMIN:-0.001}             # Dr. OPD: lower clip of w = clip(1 + lambda*z, w_min, 3)
NGPU=${NGPU:-8}

runtime_env() {
  export PATH="$VENV/bin:$PATH"
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  export HF_HOME=${HF_HOME:-$WORK/hf}
  export VLLM_CACHE_ROOT=$WORK/vllm_cache VLLM_DISABLE_COMPILE_CACHE=1
  export XDG_CACHE_HOME=$WORK/xdg TRITON_CACHE_DIR=$WORK/triton TORCHINDUCTOR_CACHE_DIR=$WORK/inductor
  export CUDA_CACHE_PATH=$WORK/nv_cache CUDA_CACHE_MAXSIZE=4294967296
  export WANDB_MODE=${WANDB_MODE:-offline} WANDB_DIR=$WORK/wandb
  export TOKENIZERS_PARALLELISM=true NCCL_DEBUG=WARN VLLM_LOGGING_LEVEL=WARN
  export RAY_USAGE_STATS_ENABLED=0 RAY_DISABLE_DOCKER_CPU_WARNING=1
  export NGPUS_PER_NODE=$NGPU
  # Respect the GPU allocation provided by the user or scheduler.
  unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
  mkdir -p "$WORK" "$RESULTS_DIR" "$WANDB_DIR" "$VLLM_CACHE_ROOT" "$CUDA_CACHE_PATH"
}

CMD=${1:-}
case "$CMD" in

setup)
  [ "$DRY_RUN" = 0 ] || { echo 'Would run scripts/install.sh (Python 3.11, dependencies and models). No action taken.'; exit 0; }
  mkdir -p "$WORK"
  VENV=$VENV WORK=$WORK bash "$HERE/scripts/install.sh"
  ;;

data)
  TRAIN=${TRAIN_DATASET:-$HERE/Dropd/datasets/deepmath-level6-train.parquet}
  VAL=${TEST_DATASET:-$HERE/Dropd/datasets/valid_final_unique.parquet}
  printf 'Train data: %s\nEvaluation data: %s\n' "$TRAIN" "$VAL"
  [ "$DRY_RUN" = 0 ] || exit 0
  if [ -x "$VENV/bin/python" ]; then DATA_PY="$VENV/bin/python"; else DATA_PY=${PYTHON:-python3}; fi
  "$DATA_PY" "$HERE/scripts/check_data.py" "$TRAIN" "$VAL"
  ;;

train)
  ARM=${2:?arm: dropd|plain|grpd|opdgrpo|exopd}
  LAM=${3:-0.4}
  SEED=${4:-2}
  [ "$#" -le 4 ] || { echo 'Too many arguments; see --help.' >&2; exit 2; }
  case "$ARM" in dropd|plain|grpd|opdgrpo|exopd) ;; *) echo "Unknown method: $ARM" >&2; exit 2 ;; esac
  [[ "$LAM" =~ ^-?[0-9]+([.][0-9]+)?$ ]] || { echo 'lambda must be numeric.' >&2; exit 2; }
  [[ "$SEED" =~ ^[0-9]+$ ]] || { echo 'data_seed must be a non-negative integer.' >&2; exit 2; }
  [[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || { echo 'STEPS must be a positive integer.' >&2; exit 2; }
  [[ "$NGPU" =~ ^[1-9][0-9]*$ ]] || { echo 'NGPU must be a positive integer.' >&2; exit 2; }
  [[ "$EVAL_STEPS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'EVAL_STEPS must be comma-separated integers.' >&2; exit 2; }
  DS=$HERE/Dropd/datasets
  TRAIN=${TRAIN_DATASET:-$DS/deepmath-level6-train.parquet}
  VAL=${TEST_DATASET:-$DS/valid_final_unique.parquet}
  # Internal scripts change directory; make relative user paths stable first.
  case "$TRAIN" in /*) ;; *) TRAIN="$PWD/$TRAIN" ;; esac
  case "$VAL" in /*) ;; *) VAL="$PWD/$VAL" ;; esac
  NAME=math_${ARM}
  if [ "$ARM" = dropd ]; then NAME=${NAME}_lam${LAM}; fi
  NAME=${NAME}_seed${SEED}
  LOG=$RESULTS_DIR/$NAME.log
  printf 'Dr. OPD launch plan\nMethod: %s\nStudent: %s\nTeacher: %s\nLambda argument (dropd): %s\nData seed: %s\nSteps: %s\nGPUs: %s\nCUDA_VISIBLE_DEVICES: %s\nCheckpoint steps: %s\nTraining data: %s\nEvaluation data: %s\nLog: %s\n' \
    "$ARM" "$STUDENT" "$TEACHER" "$LAM" "$SEED" "$STEPS" "$NGPU" "${CUDA_VISIBLE_DEVICES:-not set}" "$EVAL_STEPS" "$TRAIN" "$VAL" "$LOG"
  echo 'Evaluation: original test_freq=1000 plus final step; TEST_STEPS is not consumed by this trainer.'
  if [ "$DRY_RUN" = 1 ]; then
    printf 'Entry point: bash Dropd/arm_%s.sh\nDry run only: no directories, GPU queries, Ray, downloads or training.\n' "$ARM"
    exit 0
  fi
  [ -f "$TRAIN" ] && [ -f "$VAL" ] || { echo 'Missing prepared data. See docs/data.md and run bash run.sh data.' >&2; exit 1; }
  [ -x "$VENV/bin/python" ] || { echo "Missing environment: $VENV. Run bash run.sh setup or set VENV." >&2; exit 1; }
  runtime_env
  [ ! -e "$LOG" ] || { echo "Refusing to overwrite an existing log: $LOG. Choose a new RESULTS_DIR." >&2; exit 1; }
  cd "$HERE/Dropd"
  env ACTOR_MODEL_PATH="$STUDENT" REWARD_MODEL_PATH="$TEACHER" \
      TRAIN_DATASET="$TRAIN" TEST_DATASET="$VAL" DATA_SEED="$SEED" \
      STEPS="$STEPS" TEST_STEPS="$EVAL_STEPS" SAVE_STEPS="$EVAL_STEPS" \
      TEST_FREQ=1000 SAVE_FREQ=1000 VAL_N="${VAL_N:-16}" VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}" \
      EXPERIMENT_NAME="$NAME" GRAD_CLIP="$GRAD_CLIP" \
      FINAL_CKPT_DIR="${FINAL_CKPT_DIR:-$WORK/checkpoints/$NAME}" \
      MAX_RESP_LENGTH="$RESP_LEN" MAX_VAL_RESP_LENGTH="$VAL_LEN" MAX_PROMPT_LENGTH=1024 \
      CREDIT_LAM="$LAM" CREDIT_ACTIVE="$CREDIT_ACTIVE" CREDIT_WMIN="$CREDIT_WMIN" CREDIT_JVP_CHECK=0 \
      bash "./arm_$ARM.sh" actor_rollout_ref.rollout.dtype=bfloat16 ++reward_model.model.dtype=bfloat16 \
      2>&1 | tee "$LOG"
  ;;

*)
  sed -n '2,18p' "$0"
  exit 1
  ;;
esac

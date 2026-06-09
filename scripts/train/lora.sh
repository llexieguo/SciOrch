#!/bin/bash
# ============================================================
# Offline REINFORCE++ Training Script
# Usage: bash examples/train/rlhf/offline_reinforce/lora.sh
#
# Notes:
#   - KL-to-ref is controlled by --offline_reinforce_kl_coef.
#   - In full FT, ms-swift auto-loads a frozen ref model from --model.
#   - In LoRA FT, the trainer uses disable_adapter() so the base model
#     acts as the reference policy without loading an extra ref model.
# ============================================================

# ======================== 按需修改区 ========================

# 模型（可通过环境变量 MODEL 覆盖）
MODEL="${MODEL:-../../model/qwen3-vl-8b-instruct}"

# 数据集路径（jsonl 文件）
# 默认使用完整 PPO 数据；如需动作加权，脚本会默认切到带 sample_weight 的完整 PPO 数据。
DEFAULT_DATASET="output/v0/iteration1/msswift_export_main/msswift_ppo.jsonl"
DEFAULT_WEIGHTED_DATASET="mcts_data/v6_pruned_weight/msswift_ppo.jsonl"

# 动作加权开关：
#   默认 false：不读取 sample_weight，使用完整 PPO 数据。
#   设为 true ：默认切到带 sample_weight 的完整 PPO 数据，并读取 SAMPLE_WEIGHT_KEY 对应列。
USE_ACTION_SAMPLE_WEIGHT="${USE_ACTION_SAMPLE_WEIGHT:-false}"
if [ "${USE_ACTION_SAMPLE_WEIGHT}" = "true" ]; then
    SAMPLE_WEIGHT_KEY="${SAMPLE_WEIGHT_KEY:-sample_weight}"
    DATASET="${DATASET:-${DEFAULT_WEIGHTED_DATASET}}"
else
    SAMPLE_WEIGHT_KEY=""
    DATASET="${DATASET:-${DEFAULT_DATASET}}"
fi

# 输出目录（可通过环境变量 OUTPUT_DIR 覆盖）
OUTPUT_DIR="${OUTPUT_DIR:-output/offline_reinforce_k1}"

# GPU 设置（可通过环境变量覆盖）
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

# 训练模式: "lora" 或 "full"（可通过环境变量 TUNER_TYPE 覆盖）
TUNER_TYPE="${TUNER_TYPE:-full}"

# 训练超参（可通过同名环境变量覆盖）
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"
MAX_LENGTH="${MAX_LENGTH:-24578}"

# LoRA 参数（仅 TUNER_TYPE=lora 时生效，可通过环境变量覆盖）
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-32}"

# 离线 REINFORCE++ 专有（可通过同名环境变量覆盖）
KL_COEF="${KL_COEF:-1}"
KL_ESTIMATOR="${KL_ESTIMATOR:-k1}"  # k1 | k3 | gspo
WHITEN_ADVANTAGES="${WHITEN_ADVANTAGES:-true}"
# 用 rank-based advantage（只保留组内排序，忽略分数大小）
# true: winner=+0.5 / loser=-0.5 / tie=0; false: r - group_mean（默认）
USE_RANK_ADVANTAGE="${USE_RANK_ADVANTAGE:-false}"
# 最终参与训练/分组的标量存在 REWARD_KEY 这一列（可被组合覆盖）
REWARD_KEY="${REWARD_KEY:-expected_acc_reward}"
ANSWER_KEY="${ANSWER_KEY:-answer}"
REWARD_KEYS="${REWARD_KEYS:-expected_acc_reward}"
REWARD_WEIGHTS="${REWARD_WEIGHTS:-1.0}"
# 组合 reward（可选）：设为列名逗号分隔与权重逗号分隔，等价于
#   REWARD_KEY = 1*acc + 0.5*llm_acc + 1*llm_score
# 留空则直接用数据里已有的 REWARD_KEY 一列
# REWARD_KEYS="acc,llm_acc,llm_score"
# REWARD_WEIGHTS="1,0.5,1"

# 保存 & 日志（可通过同名环境变量覆盖）
SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"     # "epoch" 按轮保存, "steps" 按步保存
SAVE_STEPS="${SAVE_STEPS:-200}"             # 仅 SAVE_STRATEGY=steps 时生效
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
# 设为 false 才会保存 optimizer/scheduler/trainer state，支持 resume_from_checkpoint
SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-false}"
# 断点续训路径；留空表示从头训练，例如:
# RESUME_FROM_CHECKPOINT="output/offline_reinforce_v8/checkpoint-200"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
EVAL_RATIO="${EVAL_RATIO:-0.01}"

# 日志平台: "tensorboard" 或 "wandb"（可同时用: "tensorboard wandb"，可通过 REPORT_TO 覆盖）
REPORT_TO="${REPORT_TO:-wandb}"
# wandb 设置（仅 REPORT_TO 含 wandb 时生效）
export WANDB_PROJECT="${WANDB_PROJECT:-offline-reinforce-v8}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-pruned}"
# 强制离线：避免无网环境 wandb 连网超时
export WANDB_MODE=offline

echo "[offline_reinforce] dataset=${DATASET}"
echo "[offline_reinforce] use_action_sample_weight=${USE_ACTION_SAMPLE_WEIGHT}"
if [ -n "${SAMPLE_WEIGHT_KEY}" ]; then
    echo "[offline_reinforce] sample_weight_key=${SAMPLE_WEIGHT_KEY}"
fi

# ======================== 构建命令 ========================

ARGS=(
    --rlhf_type offline_reinforce
    --model "${MODEL}"
    --dataset "${DATASET}"
    --output_dir "${OUTPUT_DIR}"
    --tuner_type "${TUNER_TYPE}"
    --torch_dtype bfloat16
    --num_train_epochs ${NUM_EPOCHS}
    --per_device_train_batch_size ${BATCH_SIZE}
    --per_device_eval_batch_size ${BATCH_SIZE}
    --learning_rate ${LEARNING_RATE}
    --lr_scheduler_type cosine
    --gradient_accumulation_steps ${GRAD_ACCUM}
    --gradient_checkpointing true
    --max_length ${MAX_LENGTH}
    --warmup_ratio 0.05
    --weight_decay 0.1
    --save_strategy ${SAVE_STRATEGY}
    --save_steps ${SAVE_STEPS}
    --save_total_limit ${SAVE_TOTAL_LIMIT}
    --save_only_model ${SAVE_ONLY_MODEL}
    --logging_steps ${LOGGING_STEPS}
    --split_dataset_ratio ${EVAL_RATIO}
    --eval_strategy ${SAVE_STRATEGY}
    --eval_steps ${SAVE_STEPS}
    --dataloader_num_workers 4
    --offline_reinforce_kl_coef ${KL_COEF}
    --offline_reinforce_kl_estimator ${KL_ESTIMATOR}
    --offline_reinforce_whiten_advantages ${WHITEN_ADVANTAGES}
    --offline_reinforce_use_rank_advantage ${USE_RANK_ADVANTAGE}
    --offline_reinforce_reward_key "${REWARD_KEY}"
    --offline_reinforce_answer_key "${ANSWER_KEY}"
    --report_to ${REPORT_TO}
    # --deepspeed zero2
)

if [ -n "${REWARD_KEYS:-}" ]; then
    ARGS+=(--offline_reinforce_reward_keys "${REWARD_KEYS}")
fi
if [ -n "${REWARD_WEIGHTS:-}" ]; then
    ARGS+=(--offline_reinforce_reward_weights "${REWARD_WEIGHTS}")
fi
if [ -n "${SAMPLE_WEIGHT_KEY:-}" ]; then
    ARGS+=(--offline_reinforce_sample_weight_key "${SAMPLE_WEIGHT_KEY}")
fi
if [ -n "${RESUME_FROM_CHECKPOINT:-}" ]; then
    ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

# LoRA 模式追加参数
if [ "${TUNER_TYPE}" = "lora" ]; then
    ARGS+=(
        --lora_rank ${LORA_RANK}
        --lora_alpha ${LORA_ALPHA}
        --target_modules all-linear
    )
fi

# ======================== 启动训练 ========================

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
NPROC_PER_NODE=${NPROC_PER_NODE} \
swift rlhf "${ARGS[@]}"

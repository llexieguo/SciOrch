#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ============================================================
# Offline REINFORCE++ Training Script (PPO-clip variant)
# Usage: bash examples/train/rlhf/offline_reinforce/lora_clip.sh
#
# Notes:
#   - Policy objective can be switched to PPO-style clipped ratio by
#     setting KL_ESTIMATOR=ppo_clip.
#   - CLIPRANGE controls PPO clipping range when ppo_clip is used.
#   - In full FT, ms-swift auto-loads a frozen ref model from --model.
#   - MODEL 若为本地 checkpoint 目录且含 optimizer.pt，且未设置 RESUME_FROM_CHECKPOINT，
#     会自动 resume_from_checkpoint="$MODEL"（接上 optimizer / scheduler）。
# ============================================================

# ======================== 按需修改区 ========================

# 模型（可通过环境变量 MODEL 覆盖）
MODEL="${MODEL:-output/v4/iteration2/training/v0-20260507-222624/checkpoint-42}"

# 数据集路径（jsonl 文件）
# 默认使用完整 PPO 数据；如需动作加权，脚本会默认切到带 sample_weight 的完整 PPO 数据。
DEFAULT_DATASET="output/v4/iteration3/msswift_export/msswift_ppo_merged.jsonl"
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
OUTPUT_DIR="${OUTPUT_DIR:-output/v4/two_stage_train}"

# GPU 设置（可通过环境变量覆盖）
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

# 训练模式: "lora" 或 "full"（可通过环境变量 TUNER_TYPE 覆盖）
TUNER_TYPE="${TUNER_TYPE:-full}"

# 训练超参（可通过同名环境变量覆盖）
# NUM_EPOCHS：HuggingFace 里是「总 epoch 目标」（会参与重算 max_steps），不是「在断点之上再加几轮」。
# resume 时：global_step 从断点接上；若 本次 num_train_epochs 对应的新 max_steps <= 断点 global_step，则 0 步更新。
# 例：断点 trainer_state.json 里 epoch≈3、global_step 已满，想再训约 3 轮 → 把 NUM_EPOCHS 调到 6（或更大），而不是仍写 3。
NUM_EPOCHS="${NUM_EPOCHS:-6}"
MAX_STEPS="${MAX_STEPS:-}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-64}"
MAX_LENGTH="${MAX_LENGTH:-24578}"

# LoRA 参数（仅 TUNER_TYPE=lora 时生效，可通过环境变量覆盖）
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-32}"

# 离线 REINFORCE++ 专有（可通过同名环境变量覆盖）
# PPO-clip 变体默认: KL_ESTIMATOR=ppo_clip, KL_COEF=0
KL_COEF="${KL_COEF:-0.02}"
KL_ESTIMATOR="${KL_ESTIMATOR:-ppo_clip}"  # k1 | k3 | gspo | ppo_clip
CLIPRANGE="${CLIPRANGE:-0.2}"             # only used when KL_ESTIMATOR=ppo_clip
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
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
# 设为 false 才会保存 optimizer/scheduler/trainer state，支持 resume_from_checkpoint
SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-false}"
LOAD_BEST_MODEL_AT_END="${LOAD_BEST_MODEL_AT_END:-false}"
# 断点续训；留空时若 MODEL 目录下有 optimizer.pt 则自动设为 MODEL（接上训练）
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
# 设为 true 则只加载权重（需配合手动 RESUME_FROM_CHECKPOINT 或上面自动 resume）
RESUME_ONLY_MODEL="${RESUME_ONLY_MODEL:-false}"
if [ -z "${RESUME_FROM_CHECKPOINT}" ] && [ -d "${MODEL}" ] && [ -f "${MODEL}/optimizer.pt" ] && [ -f "${MODEL}/trainer_state.json" ]; then
    RESUME_FROM_CHECKPOINT="${MODEL}"
    echo "[offline_reinforce] MODEL contains optimizer.pt -> resume_from_checkpoint=${MODEL}"
fi
LOGGING_STEPS="${LOGGING_STEPS:-5}"
EVAL_RATIO="${EVAL_RATIO:-0.01}"

# 日志平台: "tensorboard" 或 "wandb"（可同时用: "tensorboard wandb"，可通过 REPORT_TO 覆盖）
REPORT_TO="${REPORT_TO:-wandb}"
# wandb 设置（仅 REPORT_TO 含 wandb 时生效）
export WANDB_PROJECT="${WANDB_PROJECT:-offline-reinforce-v8}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-pruned}"
# 强制离线：避免集群无外网时 wandb ConnectTimeout / 重试占用时间（需要云端时再改此处或改用 REPORT_TO=tensorboard）
export WANDB_MODE=offline

# 日志里显示的 epoch 是「从任务开始累计」的小数（含断点之前），不是「本次 shell 从 1 开始数」；要看本次多训几轮，用断点 epoch 与上面 NUM_EPOCHS 的差来估。
if [ -n "${RESUME_FROM_CHECKPOINT:-}" ] && [ -f "${RESUME_FROM_CHECKPOINT}/trainer_state.json" ]; then
    echo "[offline_reinforce] --- resume 断点 (trainer_state.json) ---"
    RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT}" NUM_EPOCHS="${NUM_EPOCHS}" python3 <<'PY'
import json, os
ckpt = os.environ["RESUME_FROM_CHECKPOINT"]
num_raw = os.environ.get("NUM_EPOCHS", "")
path = os.path.join(ckpt, "trainer_state.json")
with open(path, encoding="utf-8") as f:
    st = json.load(f)
ep = float(st.get("epoch", 0.0))
gs = int(st.get("global_step", 0))
ms = st.get("max_steps")
try:
    n_arg = float(num_raw)
except ValueError:
    n_arg = None
line = f"  epoch≈{ep:.6f}, global_step={gs}"
if ms is not None:
    line += f", 断点里记录的 max_steps={ms}"
print(line)
print("  本次训练日志里的 epoch 会从上面这个累计值继续涨（不是从 0 重新记）。")
print("  num_train_epochs 表示本次进程参数下的总目标轮数；多训须让该目标 > 断点已完成的 epoch（同数据同 batch 时通常把 NUM_EPOCHS 设为「已训满轮数 + 想再加的轮数」）。")
if n_arg is not None and n_arg <= ep + 1e-5:
    print(f"  ⚠ NUM_EPOCHS={num_raw} ≤ 断点 epoch≈{ep:.4f} 时，很可能 max_steps 已达标 → 实际 0 optimizer step（秒结束）。")
PY
    echo "[offline_reinforce] -------------------------------------------"
fi

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
    --load_best_model_at_end ${LOAD_BEST_MODEL_AT_END}
    --save_only_model ${SAVE_ONLY_MODEL}
    --logging_steps ${LOGGING_STEPS}
    --split_dataset_ratio ${EVAL_RATIO}
    --eval_strategy ${SAVE_STRATEGY}
    --eval_steps ${SAVE_STEPS}
    --dataloader_num_workers 4
    --offline_reinforce_kl_coef ${KL_COEF}
    --offline_reinforce_kl_estimator ${KL_ESTIMATOR}
    --offline_reinforce_cliprange ${CLIPRANGE}
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
if [ "${RESUME_ONLY_MODEL}" = "true" ]; then
    ARGS+=(--resume_only_model true)
fi
if [ -n "${MAX_STEPS}" ]; then
    ARGS+=(--max_steps "${MAX_STEPS}")
    echo "[lora_clip.sh] Using --max_steps ${MAX_STEPS} (overrides --num_train_epochs)"
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

#!/bin/bash
#
# Ultra-v3 MTP training — parameterized launcher
#
# This script is called by launch-script/run_experiments.py via sbatch.
# Job name and node count are set by the launcher; MTP/training params come
# from env vars (see launch-script/ultra/config.yaml).
#
# Architecture: Mamba-2 hybrid MoE, 108 layers, 512 experts, TP=8, EP=16
# Precision: BF16 (no FP8)
# Launch: bindpcie, srun + container

#SBATCH -p batch
#SBATCH -q normal
#SBATCH --account=coreai_nvfm_llm
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=8
#SBATCH --time=4:00:00
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --dependency=singleton

################################################################
### TransformerEngine
################################################################
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NVTE_CPU_OFFLOAD_V1=1
export TORCHINDUCTOR_WORKER_START=fork

################################################################
### UCX (prevents memory hook conflicts in multi-node)
################################################################
export UCX_MEM_MMAP_HOOK_MODE=none
export UCX_MEM_CUDA_HOOK_MODE=none
export UCX_MEM_MALLOC_HOOKS=n
export UCX_ERROR_SIGNALS=

################################################################
### General
################################################################
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16
export HF_HOME="/lustre/fsw/portfolios/coreai/users/$USER/.cache/huggingface/"
export TRITON_CACHE_DIR="/tmp/triton-cache"

################################################################
### Env var defaults — overridden by launcher via sbatch --export
################################################################
MTP_NUM_LAYERS=${MTP_NUM_LAYERS:-3}
MTP_KD_ENABLED=${MTP_KD_ENABLED:-true}
MTP_KD_TEMPERATURE=${MTP_KD_TEMPERATURE:-2.0}
MTP_KD_LOSS_WEIGHT=${MTP_KD_LOSS_WEIGHT:-1.0}
MTP_DISABLE_CE_LOSS=${MTP_DISABLE_CE_LOSS:-true}
MTP_HSM_MODE=${MTP_HSM_MODE:-uniform_layer_sample}
LR=${LR:-1e-5}
SEQ_LEN=${SEQ_LEN:-8192}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
LR_DECAY_STYLE=${LR_DECAY_STYLE:-cosine}
PROMPT_FORMAT=${PROMPT_FORMAT:-identity}
BLEND_PATH=${BLEND_PATH:-"/lustre/fsw/portfolios/llmservice/users/soumyes/sft-runs/blends/ultra_mar12_3pc_trunc_510k-filtered.part0.json"}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-"/lustre/fs1/portfolios/coreai/projects/coreai_nvfm_llm/users/ygalron/super_mtp_training/pretrained_ckpts/rl_step_96"}
MEGATRON_LM_DIR=${MEGATRON_LM_DIR:-"/lustre/fs1/portfolios/coreai/projects/coreai_nvfm_llm/users/$USER/super_mtp_training/megatron-lm-ultra-sft"}

################################################################
### Paths
################################################################
NAME=${SLURM_JOB_NAME}

OUTPUT_ROOT="/lustre/fs1/portfolios/coreai/projects/coreai_nvfm_llm/users/$USER/super_mtp_training/"
IMAGE="/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/adithyare/containers/pt_ultra_mamba_ssmv230_23jan28.sqsh"
BINDPCIE_SCRIPT="/lustre/fsw/portfolios/llmservice/users/soumyes/sft-runs/code/bindpcie.sh"
TOKENIZER_MODEL_PATH="/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/adithyare/nemotron_super/tokenizer"

export WANDB_API_KEY="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"
WANDB_PROJECT="ultra-v3-sft-$USER"

RUN_DIR="${OUTPUT_ROOT}"
LOGS_DIR="${RUN_DIR}/logs/${NAME}/"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints/${NAME}"
DATACACHE_DIR="${RUN_DIR}/data_cache/${NAME}"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard/${NAME}"

mkdir -p ${LOGS_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${DATACACHE_DIR}
mkdir -p ${TENSORBOARD_DIR}

################################################################
### Log environment
################################################################
DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
if [ -n "${SLURM_JOB_ID:-}" ] ; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
    ENV_LOG_FILENAME=${NAME}_${SLURM_JOB_ID}_${DATETIME}.env.log
else
    SCRIPT_PATH=$(realpath "$0")
    ENV_LOG_FILENAME=${NAME}_${DATETIME}.env.log
fi
SCRIPT_DIR=$(dirname ${SCRIPT_PATH})

echo "<< START PATHS >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "IMAGE=${IMAGE}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "BINDPCIE_SCRIPT=${BINDPCIE_SCRIPT}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "MEGATRON_LM_DIR=${MEGATRON_LM_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "RUN_DIR=${RUN_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "LOGS_DIR=${LOGS_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "DATACACHE_DIR=${DATACACHE_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "TENSORBOARD_DIR=${TENSORBOARD_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "SCRIPT_DIR=${SCRIPT_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END PATHS >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

echo "<< START GIT >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT LOG" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${MEGATRON_LM_DIR} log --oneline -1 |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT STATUS" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${MEGATRON_LM_DIR} status --porcelain --branch |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT DIFF" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${MEGATRON_LM_DIR} diff |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END GIT >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

echo "<< START ENV >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
env |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END ENV >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

################################################################
### Training hyperparams
################################################################
TRAIN_SAMPLES=1042136
LR_WARMUP_SAMPLES=6400
LR_DECAY_SAMPLES=524288
LOG_INTERVAL=1
GBS=64
MIN_LR=2e-6

################################################################
### Build MTP options from env vars
################################################################
MTP_OPTIONS=""
MTP_OPTIONS+=" --mtp-num-layers ${MTP_NUM_LAYERS}"
MTP_OPTIONS+=" --mtp-use-repeated-layer"
MTP_OPTIONS+=" --calculate-per-token-loss"
MTP_OPTIONS+=" --mtp-loss-scaling-factor 0.3"
MTP_OPTIONS+=" --freeze-base-model-for-mtp"

if [ "${MTP_HSM_MODE}" != "" ]; then
    MTP_OPTIONS+=" --mtp-hsm-mode ${MTP_HSM_MODE}"
fi

if [ "${MTP_DISABLE_CE_LOSS}" = "true" ]; then
    MTP_OPTIONS+=" --mtp-disable-ce-loss"
fi

if [ "${MTP_KD_ENABLED}" = "true" ]; then
    MTP_OPTIONS+=" --mtp-kd-logit-enabled"
    MTP_OPTIONS+=" --mtp-kd-logit-temperature ${MTP_KD_TEMPERATURE}"
    MTP_OPTIONS+=" --mtp-kd-logit-loss-weight ${MTP_KD_LOSS_WEIGHT}"
fi

################################################################
### Build command
################################################################
OPTIONS=" \
    --sft \
    --sft-tokenizer-prompt-format ${PROMPT_FORMAT} \
    --distributed-timeout-minutes 30 \
    --num-dataset-builder-threads 32 \
    --tokenizer-type SFTTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL_PATH} \
        --recompute-granularity selective \
        --recompute-modules moe \
        \
        --fine-grained-activation-offloading \
        --offload-modules moe_act \
        \
        --context-parallel-size 1 \
        --tensor-model-parallel-size 8 \
        --expert-model-parallel-size 16 \
        --expert-tensor-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --hybrid-override-pattern MEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME \
        --mtp-hybrid-override-pattern \"*E\" \
        \
        --pretrained-checkpoint ${BASE_MODEL_PATH} \
        --save-interval ${SAVE_INTERVAL} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style ${LR_DECAY_STYLE} \
        --train-samples ${TRAIN_SAMPLES} \
        --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
        --lr-decay-samples ${LR_DECAY_SAMPLES} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --log-interval ${LOG_INTERVAL} \
        --micro-batch-size 1 \
        --global-batch-size ${GBS} \
        --overlap-grad-reduce \
        --overlap-param-gather \
        \
        ${MTP_OPTIONS} \
        \
        --high-priority-stream-groups ep \
        --manual-gc-interval 10 \
        --ddp-num-buckets 10 \
        --manual-gc \
        \
        --moe-latent-size 2048 \
        --moe-permute-fusion \
        --cross-entropy-loss-fusion \
        --cross-entropy-fusion-impl native \
        --use-fused-weighted-squared-relu \
        \
        --moe-token-dispatcher-type alltoall \
        --moe-router-score-function sigmoid \
        --moe-grouped-gemm \
        --num-experts 512 \
        --moe-router-topk 22 \
        --moe-aux-loss-coeff 1e-4 \
        --moe-router-topk-scaling-factor 5.0 \
        --moe-router-enable-expert-bias \
        --moe-router-dtype fp32 \
        --moe-router-load-balancing-type seq_aux_loss \
        --moe-shared-expert-intermediate-size 10240 \
        \
        --attention-backend flash \
        --num-workers 1 \
        --disable-gloo-process-groups \
        --ckpt-format torch_dist \
        --ckpt-fully-parallel-save \
        --ckpt-fully-parallel-load \
        --ckpt-assume-constant-structure \
        --dist-ckpt-save-pre-mcore-014 \
        \
        --squared-relu \
        --no-mmap-bin-files \
        --exit-duration-in-mins 210 \
        --no-create-attention-mask-in-dataloader \
        \
        --sequence-parallel \
        --use-distributed-optimizer \
        --override-opt-param-scheduler \
        \
        --mamba-num-heads 256 \
        --is-hybrid-model \
        --untie-embeddings-and-output-weights \
        --init-method-std 0.0099 \
        --position-embedding-type none \
        --num-layers 108 \
        --hidden-size 8192 \
        --num-attention-heads 64 \
        --group-query-attention \
        --num-query-groups 2 \
        --ffn-hidden-size 5120 \
        --kv-channels 128 \
        --save ${CHECKPOINT_DIR} \
        --load ${CHECKPOINT_DIR} \
        --per-split-data-args-path ${BLEND_PATH} \
        --data-cache-path ${DATACACHE_DIR} \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --disable-bias-linear \
        --normalization RMSNorm \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --log-num-zeros-in-grad \
        --log-throughput \
        --log-timers-to-tensorboard \
        --log-progress \
        --log-energy \
        --log-memory-interval 200 \
        --logging-level 20 \
        --log-straggler \
        --disable-straggler-on-startup \
        --straggler-minmax-count 16 \
        --check-weight-hash-across-dp-replicas-interval 20000 \
        --ddp-pad-buckets-for-high-nccl-busbw \
        --timing-log-option minmax \
        --eval-interval 1000 \
        --eval-iters 14 \
        --bf16 \
        --use-mcore-models \
        --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
        --wandb-project ${WANDB_PROJECT} \
        --wandb-exp-name ${NAME} \
        --dist-ckpt-strictness log_unexpected \
        --tensorboard-dir ${TENSORBOARD_DIR}"

# Append extra CLI options from launcher suffixes (e.g. --mtp-share-kv)
if [ -n "${EXTRA_OPTIONS}" ]; then
    OPTIONS+=" ${EXTRA_OPTIONS}"
fi

RUN_CMD="python -u ${MEGATRON_LM_DIR}/pretrain_mamba.py ${OPTIONS}"

LAUNCH_CMD="${BINDPCIE_SCRIPT} --cpu=node --mem=node -- ${RUN_CMD}"

srun -l \
     --mpi=none \
     --no-container-mount-home \
     --container-image=${IMAGE} \
     --container-mounts="/lustre:/lustre" \
     --container-env=NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN,USE_MNNVL,UCX_MEM_MMAP_HOOK_MODE,UCX_MEM_CUDA_HOOK_MODE,UCX_MEM_MALLOC_HOOKS,UCX_ERROR_SIGNALS,NVTE_CPU_OFFLOAD_V1,NVTE_FWD_LAYERNORM_SM_MARGIN,NVTE_BWD_LAYERNORM_SM_MARGIN,TORCHINDUCTOR_WORKER_START,PYTORCH_CUDA_ALLOC_CONF,OMP_NUM_THREADS,SLURM_LOCALID \
     --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
     sh -c "${LAUNCH_CMD}"

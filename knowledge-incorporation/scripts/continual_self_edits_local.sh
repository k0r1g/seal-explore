#!/bin/bash
# Modified version for running in SSH pod environment (no SLURM, no .env needed)

# -------- Environment ------------------------------------------------ #
# Initialize conda properly in script
source /root/.bashrc
export PATH="/workspace/miniconda3/bin:$PATH"

# Initialize conda for bash (required for activation to work in script)
eval "$(/workspace/miniconda3/bin/conda shell.bash hook)"

# Now activate the environment
conda activate seal_env

# Set HuggingFace cache to a clean directory and disable redundant caches
export HF_HOME="/workspace/hf_cache"
export TRANSFORMERS_CACHE="/workspace/hf_cache"
export HF_HUB_CACHE="/workspace/hf_cache"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1
export HF_XET_DISABLE=1
export DISABLE_XET=1
mkdir -p "$HF_HOME"

# Verify environment is activated
echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python3)"

cd /workspace/seal-explore

# -------- User-editable ---------------------------------------------- #
INDEX=0  # Index for this job, used to differentiate runs

MODEL_NAME="Qwen/Qwen2.5-7B"   # initialized model. Use the last RL checkpoint
DATASET="knowledge-incorporation/data/squad_val.json"  # evaluation dataset
OUTPUT_DIR="knowledge-incorporation/results/continual_self_edits/kori/run${INDEX}"
mkdir -p "${OUTPUT_DIR}"

# LoRA / tuning hyper-parameters (matches: r α drop ep lr bs ga)
LORA_RANK=32
LORA_ALPHA=64
LORA_DROPOUT=0
FINETUNE_EPOCHS=10
FINETUNE_LR=1e-3
BATCH_SIZE=1
GRAD_ACC=1

# Infrastructure layout - Using your pod's ports
VLLM_SERVER_GPUS="0"       # GPU(s) for vLLM server (comma-sep)
PY_DRIVER_GPU="1"          # GPU on which the continual self-edit script runs
VLLM_PORT=${VLLM_PORT:-8002}  # Changed from 8001 to 8002
ZMQ_PORT=5555              # ZMQ port (your pod uses 5555)
SEED=$((42 + INDEX))

MAX_TOKENS=2048            # reduced from 8192 to save memory
TEMPERATURE=1.0            # self-edit sampling temperature
top_p=0.95                 # self-edit top-p
GPU_MEMORY_UTILIZATION=0.8 # GPU memory utilization (0.8 = 80%)

# Reduced for testing - you can increase these later
N_SEQUENCES=2              # number of sequence to average over (reduced from 8)
N_DATAPOINTS=4             # datapoints per sequence (reduced from 8)
# --------------------------------------------------------------------- #

export CUDA_VISIBLE_DEVICES=${PY_DRIVER_GPU},${VLLM_SERVER_GPUS}

# Create logs directory
mkdir -p logs

# -------- Clean up any lingering processes --------------------------- #
echo "Cleaning up any lingering processes on ports ${VLLM_PORT} and ${ZMQ_PORT}..."
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "TTT_server" 2>/dev/null || true
fuser -k ${VLLM_PORT}/tcp 2>/dev/null || true
fuser -k ${ZMQ_PORT}/tcp 2>/dev/null || true
sleep 2

# -------- Launch Driver ---------------------------------------------- #
echo "Starting continual self-edits driver on GPU ${PY_DRIVER_GPU}"
echo "vLLM will use GPU ${VLLM_SERVER_GPUS} on port ${VLLM_PORT}"
echo "ZMQ will use port ${ZMQ_PORT}"
echo "Working directory: $(pwd)"
echo "Dataset: ${DATASET}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Available disk space: $(df -h /workspace | tail -1 | awk '{print $4}')"
echo "HuggingFace cache: ${HF_HOME}"
echo "GPU memory utilization: ${GPU_MEMORY_UTILIZATION}"

python3 -u -m knowledge-incorporation.src.continual.continual_self_edits \
    --dataset "${DATASET}" \
    --model "${MODEL_NAME}" \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --finetune_epochs ${FINETUNE_EPOCHS} \
    --finetune_lr ${FINETUNE_LR} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --n_sequences ${N_SEQUENCES} \
    --n_datapoints ${N_DATAPOINTS} \
    --output_dir "${OUTPUT_DIR}" \
    --gpus "${VLLM_SERVER_GPUS},${PY_DRIVER_GPU}" \
    --vllm_port ${VLLM_PORT} \
    --zmq_port ${ZMQ_PORT} \
    --temperature ${TEMPERATURE} \
    --top_p ${top_p} \
    --max_tokens ${MAX_TOKENS} \
    --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
    --seed ${SEED}

echo "Job finished." 
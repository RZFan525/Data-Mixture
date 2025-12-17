export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_GDR_LEVEL=4
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_IB_TC=186
export NCCL_NVLS_ENABLE=0
export NCCL_IB_GID_INDEX=3
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_TIMEOUT=22 
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_HCA=^=mlx5_3,mlx5_4,mlx5_5,mlx5_bond_0
ulimit -l unlimited

export WANDB_API_KEY=""
export WANDB_MODE="offline"

export WANDB_DIR="wandb_log"
mkdir -p $WANDB_DIR



export PATH="/inspire/hdd/global_user/liupengfei-24025/rzfan/miniconda3/bin:$PATH"
source activate factory
cd /inspire/hdd/global_user/liupengfei-24025/rzfan/LLaMA-Factory

export logging_dir=./logging
export save_name="nemotro_science_model_eval_Qwen2.5_32B_random_selection_17w.lr5e-6.bs128-epoch-9"
mkdir -p "${logging_dir}/${save_name}"

export WANDB_PROJECT="science_sft"
export WANDB_NAME=${save_name}


yaml="/inspire/hdd/global_user/liupengfei-24025/rzfan/LLaMA-Factory/yaml/qwen2.5_7B_nemotro_science_model_eval_Qwen2.5_32B_random_selection_17w.yaml"

FORCE_TORCHRUN=1 NNODES=${PET_NNODES} NODE_RANK=${PET_NODE_RANK} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} /inspire/hdd/global_user/liupengfei-24025/rzfan/miniconda3/envs/factory/bin/llamafactory-cli train $yaml > ${logging_dir}/${save_name}/${PET_NODE_RANK}_${PET_NNODES}.log 2>&1
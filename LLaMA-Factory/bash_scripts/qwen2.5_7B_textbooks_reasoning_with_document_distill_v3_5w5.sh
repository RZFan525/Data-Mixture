export WANDB_API_KEY="7ddd1c3e592ae52beef725e0192a51163dec750b"
export WANDB_MODE="offline"

export WANDB_DIR="wandb_log"
mkdir -p $WANDB_DIR



export PATH="/inspire/hdd/global_user/liupengfei-24025/rzfan/miniconda3/bin:$PATH"
source activate factory
cd /inspire/hdd/global_user/liupengfei-24025/rzfan/LLaMA-Factory

export logging_dir=./logging
export save_name=textbooks_reasoning_with_document_distill_v3_5w5.lr5e-6.bs64
mkdir -p "${logging_dir}/${save_name}"

export WANDB_PROJECT="science_sft"
export WANDB_NAME=${save_name}


yaml=/inspire/hdd/global_user/liupengfei-24025/rzfan/LLaMA-Factory/yaml/qwen2.5_7B_textbooks_reasoning_with_document_distill_v3_5w5.yaml

/inspire/hdd/global_user/liupengfei-24025/rzfan/miniconda3/envs/factory/bin/llamafactory-cli train $yaml > ${logging_dir}/${save_name}/log.log 2>&1
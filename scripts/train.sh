sbatch <<EOT
#!/bin/bash
#SBATCH --partition=mit_normal_gpu       # Partition name
#SBATCH --gres=gpu:h200:2              # Number of GPUs
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export WANDB_PROJECT=GPTeacher

cd ../LLaMA-Factory
echo "Training Qwen3.0-6B Adversarial..."
llamafactory-cli train ../configs/qwen3_0-6B_adversarial.yaml
EOT
#!/bin/bash
# One variant on one benchmark (resumable).
#
#   sbatch rebuttal_experiments/slurm/run.sh <benchmark> <variant>
#   e.g. sbatch rebuttal_experiments/slurm/run.sh licn_heldout full_classify
#
# benchmarks: artpedia_vqa aqua artquest licn_heldout artcurate_aic
# Gemma 3 variants (base_gemma3, full_gemma3) need two GPUs: add --gres=gpu:2.

#SBATCH --account=<your_account>
#SBATCH --partition=<your_gpu_partition>
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --job-name=artseek-run
#SBATCH --output=%x-%j.out

source rebuttal_experiments/slurm/env.sh
benchmark_config "$1"
VARIANT="$2"
[[ "$VARIANT" == base* ]] || use_qdrant

python rebuttal_experiments/common/run_experiment.py \
    --variant "$VARIANT" --vqa-path "$VQA" --out-dir "$BENCH_DIR/results"

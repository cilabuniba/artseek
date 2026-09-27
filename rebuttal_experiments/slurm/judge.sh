#!/bin/bash
# Judge the answers of one benchmark with phi-4 (only what is not judged yet).
#
#   sbatch rebuttal_experiments/slurm/judge.sh <benchmark> <variant> [<variant> ...]

#SBATCH --account=<your_account>
#SBATCH --partition=<your_gpu_partition>
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --job-name=artseek-judge
#SBATCH --output=%x-%j.out

source rebuttal_experiments/slurm/env.sh
benchmark_config "$1"; shift
ARGS=(); for v in "$@"; do ARGS+=(--variant "$v"); done

python rebuttal_experiments/common/analysis/build_runs_jsonl.py
python rebuttal_experiments/common/judge.py --pass answer --rubric "$RUBRIC" "${ARGS[@]}"

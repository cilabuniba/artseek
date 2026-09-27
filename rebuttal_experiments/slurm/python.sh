#!/bin/bash
# Any other GPU script (probes, evidence judge, benchmark builders, human study).
#
#   sbatch rebuttal_experiments/slurm/python.sh [<benchmark>] <script.py> [args ...]
#   e.g. sbatch rebuttal_experiments/slurm/python.sh artpedia_vqa rebuttal_experiments/common/oracle_probe.py
#
# With a benchmark name first, ARTSEEK_EXP_DIR / ARTSEEK_IMAGES_DIR are set for
# it. QDRANT_URL is set when the Qdrant server is running.

#SBATCH --account=<your_account>
#SBATCH --partition=<your_gpu_partition>
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --job-name=artseek-py
#SBATCH --output=%x-%j.out

source rebuttal_experiments/slurm/env.sh
if [[ "${1:-}" != *.py ]]; then benchmark_config "$1"; shift; fi
[ -f rebuttal_experiments/.qdrant_host ] && use_qdrant
python "$@"

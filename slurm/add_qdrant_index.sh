#!/bin/bash
# Build the HNSW index of the collection (~8 hours). CPU only.
#
#   QDRANT_BIN=/path/to/qdrant QDRANT_STORAGE=/path/to/qdrant_storage \
#       sbatch slurm/add_qdrant_index.sh

#SBATCH --account=<your_account>
#SBATCH --partition=<your_partition>
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --job-name=artseek-index
#SBATCH --output=%x-%j.out

set -euo pipefail
source .venv/bin/activate
set -a; [ -f .env ] && source .env; set +a
export QDRANT_URL=http://localhost

bash scripts/start_qdrant.sh
python -m artseek.data.main add-qdrant-index
pkill -f "$QDRANT_BIN" || true

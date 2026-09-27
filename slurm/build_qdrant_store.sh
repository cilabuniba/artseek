#!/bin/bash
# Ingest the WikiFragments embeddings into Qdrant with 4 parallel processes
# (~2.5 hours). Then run slurm/add_qdrant_index.sh.
#
# Fill in the #SBATCH placeholders, then from the repository root:
#   QDRANT_BIN=/path/to/qdrant QDRANT_STORAGE=/path/to/qdrant_storage \
#       sbatch slurm/build_qdrant_store.sh

#SBATCH --account=<your_account>
#SBATCH --partition=<your_partition>
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --job-name=artseek-store
#SBATCH --output=%x-%j.out

set -euo pipefail
# Load whatever your cluster needs here (e.g. `module load ...`).

source .venv/bin/activate
set -a; [ -f .env ] && source .env; set +a
export QDRANT_URL=http://localhost

bash scripts/start_qdrant.sh

for i in 0 1 2 3; do
    srun -u --exclusive --ntasks=1 \
        python -m artseek.data.main make-qdrant-store --process-idx "$i" --num-proc 4 &
done
wait

echo "Ingestion finished, stopping Qdrant."
pkill -f "$QDRANT_BIN" || true

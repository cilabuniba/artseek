#!/bin/bash
# Long-running Qdrant server shared by the retrieval jobs (CPU only). It writes
# its host name to rebuttal_experiments/.qdrant_host. Wait until it answers
# (curl http://<host>:6333/collections) before submitting retrieval jobs.
#
#   QDRANT_BIN=/path/to/qdrant QDRANT_STORAGE=/path/to/qdrant_storage \
#       sbatch rebuttal_experiments/slurm/qdrant_server.sh

#SBATCH --account=<your_account>
#SBATCH --partition=<your_partition>
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --job-name=qdrant
#SBATCH --output=%x-%j.out

source rebuttal_experiments/slurm/env.sh
# Listen on all interfaces: the jobs run on other nodes.
export QDRANT__SERVICE__HOST=0.0.0.0
hostname > rebuttal_experiments/.qdrant_host
mkdir -p "$QDRANT_STORAGE"
export QDRANT__STORAGE__STORAGE_PATH="$QDRANT_STORAGE"
export QDRANT__STORAGE__SNAPSHOTS_PATH="$QDRANT_STORAGE/../qdrant_snapshots"
exec "$QDRANT_BIN"

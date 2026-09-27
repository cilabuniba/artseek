#!/bin/bash
# Run the ArtSeek demo (Qdrant + Streamlit) on a GPU node.
#
#   QDRANT_BIN=/path/to/qdrant QDRANT_STORAGE=/path/to/qdrant_storage \
#       sbatch slurm/demo.sh
#
# The log prints the SSH command that forwards the demo to your machine.

#SBATCH --account=<your_account>
#SBATCH --partition=<your_gpu_partition>
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --job-name=artseek-demo
#SBATCH --output=%x-%j.out

set -euo pipefail
source .venv/bin/activate
set -a; [ -f .env ] && source .env; set +a
export QDRANT_URL=http://localhost
PORT=${PORT:-8501}

bash scripts/start_qdrant.sh

echo "On your machine run:  ssh -N -L ${PORT}:$(hostname):${PORT} ${USER}@<login_node>"
echo "then open http://localhost:${PORT}. The model loads when the first image is uploaded (a few minutes)."
streamlit run app.py --server.port "$PORT" --server.address 0.0.0.0 --server.headless true

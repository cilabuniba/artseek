# Shared setup, sourced by the job scripts (run from the repository root).
set -euo pipefail
# Load whatever your cluster needs here (e.g. `module load ...`).
source .venv/bin/activate
set -a; [ -f .env ] && source .env; set +a
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

# Benchmark folder, VQA file, image folder and judge rubric.
benchmark_config() {
    BENCH_DIR="rebuttal_experiments/$1"
    case "$1" in
        artpedia_vqa)  VQA="$BENCH_DIR/data/artpedia_vqa.json";  IMAGES="data/external/artpedia/images";      RUBRIC=artpedia ;;
        aqua)          VQA="$BENCH_DIR/data/aqua_vqa.json";      IMAGES="data/external/semart/images";        RUBRIC=aqua ;;
        artquest)      VQA="$BENCH_DIR/data/artquest_vqa.json";  IMAGES="data/external/semart/images";        RUBRIC=aqua ;;
        licn_heldout)  VQA="$BENCH_DIR/data/licn_vqa.json";      IMAGES="data/external/licn_heldout/images";  RUBRIC=aqua ;;
        artcurate_aic) VQA="$BENCH_DIR/data/artcurate_vqa.json"; IMAGES="data/external/artcurate_aic/images"; RUBRIC=aqua ;;
        *) echo "unknown benchmark: $1" >&2; exit 1 ;;
    esac
    export ARTSEEK_EXP_DIR="$PWD/$BENCH_DIR" ARTSEEK_IMAGES_DIR="$PWD/$IMAGES"
}

# Point QDRANT_URL at the server started by qdrant_server.sh.
use_qdrant() {
    local host_file=rebuttal_experiments/.qdrant_host
    [ -f "$host_file" ] || { echo "start slurm/qdrant_server.sh first" >&2; exit 1; }
    export QDRANT_URL="http://$(cat "$host_file")"
    echo "Qdrant at $QDRANT_URL"
}

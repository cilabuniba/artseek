#!/bin/bash
# Start a Qdrant server from a release binary (no Docker needed, e.g. on HPC).
#
#   QDRANT_BIN=/path/to/qdrant QDRANT_STORAGE=/path/to/qdrant_storage \
#       bash scripts/start_qdrant.sh
#
# Get the binary from https://github.com/qdrant/qdrant/releases (v1.17.0) or
# build it from source (see README). The server listens on 6333 (HTTP) and 6334
# (gRPC, used by ArtSeek). The script returns once the server answers; the
# server keeps running in the background and logs to $QDRANT_STORAGE/../qdrant.log.

set -euo pipefail
: "${QDRANT_BIN:?set QDRANT_BIN to the qdrant executable}"
: "${QDRANT_STORAGE:?set QDRANT_STORAGE to the storage directory}"

mkdir -p "$QDRANT_STORAGE"
LOG="$(dirname "$QDRANT_STORAGE")/qdrant.log"

QDRANT__STORAGE__STORAGE_PATH="$QDRANT_STORAGE" \
QDRANT__STORAGE__SNAPSHOTS_PATH="$QDRANT_STORAGE/../qdrant_snapshots" \
    nohup "$QDRANT_BIN" > "$LOG" 2>&1 &
echo "Qdrant started (pid $!), logging to $LOG"

# Loading a large collection takes a while after start.
for _ in $(seq 1 120); do
    if curl -sf http://localhost:6333/collections > /dev/null; then
        echo "Qdrant is ready."
        exit 0
    fi
    sleep 5
done
echo "Qdrant did not answer within 10 minutes, check $LOG" >&2
exit 1

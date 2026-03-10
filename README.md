# ArtSeek

Official code for ["ArtSeek: Deep artwork understanding via multimodal in-context reasoning and late interaction retrieval"](https://www.arxiv.org/abs/2507.21917).

![ArtSeek pipeline overview](assets/artseek.png)

## Summary

ArtSeek is a multimodal system for understanding artworks built on three components:

1. **Multimodal Retrieval** — Retrieves relevant information from 5M+ multimodal fragments of Wikipedia's visual arts section via a late interaction mechanism built on ColQwen2.
2. **Late Interaction Classification Network (LICN)** — Predicts artwork attributes (artist, genre, style) by combining multimodal retrieval with a multi-head classifier.
3. **In-Context Reasoning** — A Qwen2.5-VL-32B MLLM reasons over retrieved fragments and predicted attributes to answer open-ended questions about artworks.

## Installation

> [!NOTE]
> Developed on: 8 CPUs, 1× NVIDIA A100 (CUDA 12.6), 128 GB RAM.

```bash
uv sync
uv pip install flash-attn --no-build-isolation
```

## Qdrant Setup

ArtSeek uses a Qdrant vector store. On HPC systems, build Qdrant from source:
```bash
# Requires Rust 1.87.0+, LLVM/Clang (e.g. via Spack)
spack load gcc
spack load llvm
export LLVM_ROOT=$(spack location -i llvm)
export LIBCLANG_PATH=$LLVM_ROOT/lib

# Install Protobuf locally
mkdir -p ./proto_bin
curl -LO https://github.com/protocolbuffers/protobuf/releases/download/v25.1/protoc-25.1-linux-x86_64.zip
unzip -o protoc-25.1-linux-x86_64.zip -d ./proto_bin
export PROTOC=$(pwd)/proto_bin/bin/protoc
export PATH=$(pwd)/proto_bin/bin:$PATH

# Build Qdrant
git clone https://github.com/qdrant/qdrant.git
cd qdrant && git checkout v1.17.0
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin qdrant
```

## Try ArtSeek

> [!NOTE]
> Running the full pipeline requires the Qdrant server with the `wikifragments-visual-arts-embeds` collection loaded. The store requires ~250 GB of disk space.

### 1. Download datasets and models

Download the embeddings dataset from Hugging Face:
```python
from datasets import load_dataset
ds = load_dataset("cilabuniba/wikifragments-visual-arts-embeds", num_proc=4)
```

Download the LICN classifier:
```bash
hf download cilabuniba/artseek-licn
```

### 2. Build the Qdrant store

Start your Qdrant server, then ingest the embeddings. We provide `launches/make_qdrant_store.sh` to reproduce our setup (4 GPUs, 8 CPUs per process):
```bash
python -m artseek.data.main make-qdrant-store --process-idx 0 --num-proc 1
```

Increase `--num-proc` and run multiple processes in parallel (index 0 to N-1) to speed up ingestion (~2.5 hours with 4 processes).

### 3. Add the Qdrant index
```bash
python -m artseek.data.main add-qdrant-index
```

> ⏱ This step takes approximately **8 hours**.

### 4. Configure the model path

In `artseek/method/generate/pipe.py`, update the `licn_pretrained_path` with your local snapshot path from the downloaded `artseek-licn` model:
```python
MODEL = Qwen2_5_VLRAGModel(
    retriever_pretrained_model_name_or_path=get_models_dir() / "colqwen2-v1.0",
    retriever_collection_name="wikifragments-visual-arts-embeds",
    model_pretrained_model_name_or_path="Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
    licn_pretrained_path="data_/hf/hub/models--cilabuniba--artseek-licn/snapshots/<your-snapshot-hash>",
)
```

### 5. Run the notebook

Open `try.ipynb` to interact with ArtSeek. You can toggle classification and retrieval via the `classify` and `retrieve` parameters in `build_graph`.

## Train the LICN Module
```bash
accelerate launch -m artseek.method.classify.train_li_classification_network train \
  --config-path models/configs/classify/li_classification_network_tft.yaml
```

## Evaluation

### Retrieval
Run `artseek/method/retrieve/eval.py`. Create separate data stores for each configuration to evaluate.

### Classification
```bash
accelerate launch -m artseek.method.classify.train_li_classification_network test \
  --config-path models/configs/classify/li_classification_network_tft.yaml
```

### Text Generation
```bash
# Run inference
python -m artseek.method.generate.test inference \
  --config-path models/configs/generate/artpedia_short.yaml

# Compute NLP metrics
python -m artseek.method.generate.eval pred-message-to-str \
  --config-path models/configs/generate/artpedia_short.yaml
```

> Note: consider disabling SPICE for large datasets like PaintingForm. See `data/README.md` for dataset setup instructions (ArtPedia, PaintingForm, SemArt v2.0).

---

## Reproducibility

The sections below document the original data pipeline for full reproducibility. **These steps are not needed to try ArtSeek** — all datasets and models are available on Hugging Face.

### Neo4j (ArtGraph)

1. Install [Neo4j Community Edition 4.4.47](https://neo4j.com/download-center/).
2. Extract: `tar -xzf neo4j-community-4.4.47-unix.tar.gz`
3. Download the [ArtGraph dump](https://zenodo.org/records/8172374) (`artgraph2.0.dump`).
4. Load the graph: `./neo4j-community-4.4.47/bin/neo4j-admin load --from=artgraph2.0.dump --database=neo4j --force`
5. Download the [APOC JAR (v4.4.0.24)](https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/4.4.0.24) and place it in the `plugins/` folder.
6. Enable APOC procedures in `conf/neo4j.conf`.

### Data Pipeline (in order)

All steps are run via `artseek.data.main`:

1. **`download-wikiart-images`** — Downloads WikiArt images (target: 116,475 images). Repeat until complete.
2. **`make-artgraph-dataset`** — Builds the multitask classification dataset from Neo4j. Requires ArtGraph running.
3. **`define-valid-labels-artgraph-dataset`** — Defines evaluation labels.
4. **`get-visual-arts-dataset-pages`** — Recursively collects Wikipedia pages under "Visual arts" (depth=5).
5. **WikiExtractor** (run from `wikiextractor/` directory):
    ```bash
    python WikiExtractorNew.py --json -s --lists --links \
        ../data/dumps/enwiki-latest-pages-articles.xml.bz2 -o text_en
    ```
   This is a modified WikiExtractor that preserves image URLs and captions as `<a>` tags.

6. **`download-and-save-images-wikipedia`** — Downloads images from extracted Wikipedia pages.
7. **`create-wikifragments-dataset`** — Builds an HF dataset of Wikipedia paragraphs with attached images.
8. **`create-wikifragments-visual-arts-full-dataset`** — Filters to visual arts pages and builds fragment images.
9. **`colqwen-embed-new`** — Embeds fragments using ColQwen2 multi-vector representations (parallelizable).
10. **`make-qdrant-store`** — Ingests embeddings into Qdrant.
11. **`add-qdrant-index`** — Adds the HNSW index for efficient retrieval.
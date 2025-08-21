# Vector DB Benchmark for Music Semantic Search

> **This project is part of the Vector Database Benchmarking video:**
> [https://youtu.be/X0PwwfcGSHU](https://youtu.be/X0PwwfcGSHU)

This repository benchmarks multiple vector databases for music semantic search, using a shared dataset and query set. It provides both a CLI benchmarking tool and a web UI for side-by-side DB comparison.

## Features

- **Benchmarks ingest time, query latency, recall, and hit rate** for top-k search
- **Supports Qdrant, Milvus, Weaviate, Pinecone, TopK, and SQLite** (local or cloud)
- **Flexible embedding**: Use `sentence-transformers` (default) or OpenAI embeddings
- **Heuristic relevance**: Weak label matching using tags/genres for recall/hit metrics
- **Rich CLI**: Many flags for DB selection, concurrency, top-k sweep, teardown, etc.
- **Modern UI**: FastAPI backend + static frontend for live DB comparison
- **Automated result plots**: Generates summary charts and per-k metrics tables

---

## Supported Databases

- Qdrant (local/cloud)
- Milvus (local)
- Weaviate (local/cloud)
- Pinecone (local/cloud)
- TopK (cloud)

## Dataset

Use the [Muse Musical Sentiment dataset](https://www.kaggle.com/datasets/cakiki/muse-the-musical-sentiment-dataset) from Kaggle. Place the CSV as `data/muse.csv`.

You can test with `data/sample_data.csv` for a dry run.

---

## Quick Start

1. **Install dependencies**

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2. **Configure environment**

- Copy `.env.example` to `.env` and fill in DB URLs/API keys as needed

3. **Start local DBs (optional)**

```sh
docker compose -f scripts/docker-compose.yml up -d
```

4. **Generate embeddings**

```sh
python embeddings/embed.py --csv data/muse.csv --out data/embeddings.parquet
# For OpenAI: add --use_openai [--model text-embedding-3-large]
```

5. **Run the benchmark**

```sh
python benchmark.py --csv data/muse.csv --embeddings data/embeddings.parquet --dbs qdrant milvus weaviate pinecone topk sqlite --topk 10 --repetitions 5
# See all CLI flags with: python benchmark.py --help
```

6. **View results**

- Summary and per-k plots: `results/`
- Metrics: `results/metrics.json`

---

## CLI Usage

```sh
python benchmark.py --csv data/muse.csv --embeddings data/embeddings.parquet --dbs qdrant milvus weaviate pinecone topk sqlite --topk 10 --repetitions 5 [--teardown_after_benchmark]
```

**Key flags:**

- `--dbs`: List of DBs to benchmark (qdrant, milvus, weaviate, pinecone, topk, sqlite)
- `--topk`: Top-k for search (default: 10)
- `--topk_sweep`: List of k values to sweep (e.g. 5 10 50)
- `--repetitions`: Number of repetitions per query
- `--concurrency`: Number of concurrent query workers
- `--teardown_after_benchmark`: Delete DB/index after run
- `--query_model`: Embedding model for queries
- `--queries`: Path to YAML file with queries/expected labels

**Results:**

- Plots and tables in `results/` (per-k and summary)
- All metrics in `results/metrics.json`

---

## Embedding Generation

By default, uses `sentence-transformers/all-MiniLM-L6-v2`. To use OpenAI embeddings:

```sh
python embeddings/embed.py --csv data/muse.csv --out data/embeddings.parquet --use_openai --model text-embedding-3-large
```

---

## UI: Music Semantic Search – Multi-DB Compare

The `ui/` folder provides a FastAPI backend and static frontend for live, side-by-side DB search and latency comparison.

### UI Features

- Compare Qdrant, Milvus, Weaviate, Pinecone, TopK, and SQLite in parallel
- Per-DB query latency in ms
- Simple, modern UI (HTML/JS/CSS)

### UI Quick Start

1. **Install dependencies**

```sh
pip install -r requirements.txt
```

2. **Configure**

- Create `.env` in repo root with DB endpoints and API keys

3. **Run the server**

```sh
uvicorn backend.server:app --reload --port 8000
```

4. **Open the app**

- Go to [http://localhost:8000](http://localhost:8000)

---

## Project Structure

- `benchmark.py` – Main benchmarking script (CLI)
- `embeddings/embed.py` – Embedding generation (sentence-transformers or OpenAI)
- `databases/` – DB client wrappers (Qdrant, Milvus, Weaviate, Pinecone, TopK, SQLite)
- `plot_benchmarks.py` – Plots and summary tables
- `results/` – Output metrics and plots
- `ui/` – Web UI (FastAPI backend + static frontend)
- `requirements.txt` – Python dependencies

---

## Troubleshooting

- If Docker ports conflict, edit `scripts/docker-compose.yml`
- If you see dimension mismatch errors, check embedding model and DB index size
- For OpenAI, set `OPENAI_API_KEY` in your environment
- For TopK, set API key in `.env`

---

## Acknowledgements

- [Muse Musical Sentiment dataset](https://www.kaggle.com/datasets/cakiki/muse-the-musical-sentiment-dataset)
- [sentence-transformers](https://www.sbert.net/)
- [Qdrant](https://qdrant.tech/), [Milvus](https://milvus.io/), [Weaviate](https://weaviate.io/), [Pinecone](https://www.pinecone.io/), [TopK](https://topk.io/), [sqlite-vec](https://github.com/asg017/sqlite-vec)

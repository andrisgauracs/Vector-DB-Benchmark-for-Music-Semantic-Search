import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv


from databases.qdrant_client import Qdrant
from databases.milvus_client import Milvus
from databases.weaviate_client import WeaviateDB
from databases.pinecone_client import PineconeClient
from databases.topk_client import TopKClient
from databases.sqlite_client import SQLite


# --------------------
# Utilities
# --------------------
_MODEL: Optional[Any] = None
_MODEL_NAME: Optional[str] = None
_WARMED: Dict[str, bool] = {}


def embed_query(
    q: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> List[float]:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        # Re-raise with a clear message so the API can present it nicely.
        raise ImportError(
            "sentence-transformers is not installed. Install it with 'pip install sentence-transformers torch'"
        ) from e

    # Cache the model globally to avoid reloading on every request
    global _MODEL, _MODEL_NAME
    if _MODEL is None or _MODEL_NAME != model_name:
        _MODEL = SentenceTransformer(model_name)
        _MODEL_NAME = model_name
    v = _MODEL.encode([q], normalize_embeddings=True)[0]
    return v.tolist()


def get_db(name: str) -> Any:
    name = name.lower()
    if name == "qdrant":
        return Qdrant(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    if name == "milvus":
        return Milvus(
            host=os.getenv("MILVUS_HOST", "localhost"),
            port=os.getenv("MILVUS_PORT", "19530"),
        )
    if name == "weaviate":
        return WeaviateDB(url=os.getenv("WEAVIATE_URL", "http://localhost:8080"))
    if name == "pinecone":
        client = PineconeClient()
        client.print_index_stats()
        return client
    if name == "topk":
        return TopKClient()
    if name == "sqlite":
        return SQLite(db_path=os.getenv("SQLITE_DB_PATH", "music_vectors.db"))
    raise ValueError(f"Unknown DB {name}")


# --------------------
# App
# --------------------
load_dotenv()

app = FastAPI(title="Music Semantic Search Benchmark UI")

# Allow local dev from file:// or localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    topk: int = 10
    dbs: List[str] = ["qdrant", "milvus", "weaviate", "pinecone", "topk", "sqlite"]
    model: str = "sentence-transformers/all-MiniLM-L6-v2"


class DBResult(BaseModel):
    ok: bool
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    results: Optional[List[Dict[str, Any]]] = None


class SearchResponse(BaseModel):
    query: str
    topk: int
    model: str
    by_db: Dict[str, DBResult]


_clients: Dict[str, Any] = {}
_loaded: Dict[str, bool] = {}
_dim: Optional[int] = None


@app.on_event("startup")
def _startup_warmup():
    """Preload the embedding model and optionally warm selected DBs.

    Env vars:
      UI_MODEL: override model name (default sentence-transformers/all-MiniLM-L6-v2)
      UI_WARMUP_DBS: comma list of dbs to warm (default: qdrant,milvus,weaviate)
    """
    model_name = os.getenv("UI_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    try:
        _ = embed_query("warm up", model_name=model_name)
    except Exception:
        # Ignore model warmup failures; they will surface on first request
        pass

    dbs = os.getenv("UI_WARMUP_DBS", "qdrant,milvus,weaviate,pinecone,topk,sqlite").split(",")
    qvec: Optional[List[float]] = None
    for name in [d.strip() for d in dbs if d.strip()]:
        try:
            if name not in _clients:
                _clients[name] = get_db(name)
            _ensure_collection_loaded(name)
            if qvec is None:
                # build once with the preloaded model
                qvec = embed_query("warm up", model_name=model_name)
            # one small warm-up search to prime network/index caches
            try:
                _clients[name].search(qvec, top_k=1)
            except Exception:
                pass
            _WARMED[name] = True
        except Exception:
            # Ignore warmup failures; these DBs may not be running yet
            pass


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    # Prepare query embedding (with graceful ImportError handling)
    try:
        q_vec = embed_query(req.query, model_name=req.model)
    except ImportError as e:
        # Return per-DB errors to surface in UI instead of a 500
        by_db = {name: DBResult(ok=False, error=str(e)) for name in req.dbs}
        return SearchResponse(
            query=req.query, topk=req.topk, model=req.model, by_db=by_db
        )

    by_db: Dict[str, DBResult] = {}

    for name in req.dbs:
        try:
            if name not in _clients:
                _clients[name] = get_db(name)
            # Only connect and search; do not ingest or upsert any data

            # Ensure collection is loaded and warm once per DB
            if not _WARMED.get(name):
                try:
                    _ensure_collection_loaded(name)
                    _clients[name].search(q_vec, top_k=1)  # warm-up search
                except Exception as warm_e:
                    print(f"[WARN] Warmup failed for {name}: {warm_e}")
                _WARMED[name] = True

            s0 = time.time()
            matches = _clients[name].search(q_vec, top_k=req.topk)
            latency_ms = (time.time() - s0) * 1000.0

            # Normalize result shape to basic payload + score
            normalized: List[Dict[str, Any]] = []
            for r in matches:
                payload = r.get("payload") if isinstance(r, dict) else None
                score = r.get("score") if isinstance(r, dict) else None
                if payload is None and hasattr(r, "payload"):
                    payload = getattr(r, "payload")
                if score is None and hasattr(r, "score"):
                    score = getattr(r, "score")
                normalized.append(
                    {
                        "score": score,
                        "track": payload.get("track") if payload else None,
                        "artist": payload.get("artist") if payload else None,
                        "genre": payload.get("genre") if payload else None,
                        "seeds": payload.get("seeds") if payload else None,
                        "text": payload.get("text") if payload else None,
                        "payload": payload,
                    }
                )

            by_db[name] = DBResult(ok=True, latency_ms=latency_ms, results=normalized)
        except Exception as e:
            print(f"[ERROR] Search failed for {name}: {e}")
            by_db[name] = DBResult(ok=False, error=str(e), results=[])
    # Ensure all requested DBs are present in by_db, even if missing
    for name in req.dbs:
        if name not in by_db:
            by_db[name] = DBResult(ok=False, error="No results", results=[])

    # Normalize all by_db keys to lowercase for frontend compatibility
    by_db_lower = {k.lower(): v for k, v in by_db.items()}
    return SearchResponse(
        query=req.query, topk=req.topk, model=req.model, by_db=by_db_lower
    )


def _ensure_collection_loaded(db_name: str):
    """If we didn't ingest this run, try to attach to an existing
    collection/index so .search() works. Raise a helpful error if missing."""
    db = _clients[db_name]

    # Already attached?
    if hasattr(db, "col") and getattr(db, "col", None) is not None:
        return

    # Milvus: attach to existing collection if present
    if isinstance(db, Milvus):
        try:
            from pymilvus import connections, utility, Collection

            connections.connect(alias="default", host=db.host, port=db.port)
            if utility.has_collection(db.collection_name):
                db.col = Collection(db.collection_name)
                db.col.load()
                return
            raise RuntimeError(
                "Milvus collection "
                f"'{getattr(db, 'collection_name', 'music')}' not found. "
                "Either set EMBEDDINGS_PARQUET to auto-ingest or create/load the collection beforehand."
            )
        except Exception as e:
            raise RuntimeError(f"Milvus not ready for search: {e}")

    # Weaviate: attach if collection exists
    if isinstance(db, WeaviateDB):
        try:
            if db.client.collections.exists(db.class_name):
                db.col = db.client.collections.get(db.class_name)
                return
            raise RuntimeError(
                f"Weaviate collection '{db.class_name}' not found. "
                "Set EMBEDDINGS_PARQUET to auto-ingest or create it beforehand."
            )
        except Exception as e:
            raise RuntimeError(f"Weaviate not ready for search: {e}")

    # Qdrant: optional attach (your Qdrant wrapper likely doesn’t need .col)
    if isinstance(db, Qdrant):
        # If your Qdrant wrapper keeps a collection name, you can check it here.
        # Otherwise, Qdrant.search() probably builds the query against the collection name internally.
        return


# Serve frontend statically from ./frontend
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")

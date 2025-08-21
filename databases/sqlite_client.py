import sqlite3
import json
import os
from typing import List, Dict, Any
from .base import VectorDB
import sqlite_vec


class SQLite(VectorDB):
    def __init__(self, db_path: str = "music_vectors.db", table_name: str = "music_embeddings"):
        """
        Initialize SQLite client with sqlite-vec extension.

        Args:
            db_path: Path to the SQLite database file
            table_name: Name of the virtual table to create for vectors
        """
        self.db_path = db_path
        self.table_name = table_name
        self.conn = None
        self.dim = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with sqlite-vec extension loaded."""
        if self.conn is None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)

            # Enable extension loading
            conn = sqlite3.connect(self.db_path)
            conn.enable_load_extension(True)

            # Load the sqlite-vec extension
            sqlite_vec.load(conn)

            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")

            # Additional pragma settings for better performance
            conn.execute("PRAGMA synchronous=normal")
            conn.execute("PRAGMA temp_store=memory")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA legacy_alter_table=OFF")
            conn.execute("PRAGMA mmap_size=134217728")
            conn.execute("PRAGMA journal_size_limit=27103364")
            conn.execute("PRAGMA cache_size=2000")

            self.conn = conn

        return self.conn

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def setup(self, dim: int):
        """
        Set up the database with a virtual table for vector storage.

        Args:
            dim: Dimension of the vectors to be stored
        """
        self.dim = dim
        conn = self._get_connection()

        # Drop the table if it exists to start fresh
        conn.execute(f"DROP TABLE IF EXISTS {self.table_name}")

        # Create the virtual table using vec0
        conn.execute(f"""
            CREATE VIRTUAL TABLE {self.table_name} USING vec0(
                embedding float[{dim}],
                row_id INTEGER,
                track TEXT,
                artist TEXT,
                genre TEXT,
                seeds TEXT,
                text TEXT
            )
        """)

        # Note: Virtual tables cannot be indexed in sqlite-vec
        # The vector operations are handled by the vec0 extension internally

        conn.commit()

    def upsert(self, vectors: List[List[float]], payloads: List[Dict[str, Any]]):
        """
        Insert or update vectors and their metadata.

        Args:
            vectors: List of vector embeddings
            payloads: List of metadata dictionaries for each vector
        """
        if len(vectors) != len(payloads):
            raise ValueError("Number of vectors must match number of payloads")

        conn = self._get_connection()

        # Use batch inserts for better performance
        batch_size = 200
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i + batch_size]
            batch_payloads = payloads[i:i + batch_size]

            # Prepare batch insert statement
            stmt = f"""
                INSERT INTO {self.table_name}
                (rowid, embedding, row_id, track, artist, genre, seeds, text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """

            # Execute batch insert
            for j, (vector, payload) in enumerate(zip(batch_vectors, batch_payloads)):
                rowid = i + j  # Use sequential rowid
                row_id = payload.get("row_id", rowid)
                track = payload.get("track", "") or ""
                artist = payload.get("artist", "") or ""
                genre = payload.get("genre", "") or ""
                seeds = payload.get("seeds", "") or ""
                text = payload.get("text", "") or ""

                # Serialize vector as JSON for insertion
                embedding_json = json.dumps(vector)

                conn.execute(stmt, (
                    rowid,
                    embedding_json,
                    row_id,
                    track,
                    artist,
                    genre,
                    seeds,
                    text
                ))

        conn.commit()

    def search(self, query: List[float], top_k: int) -> List[Dict[str, Any]]:
        """
        Search for the top_k most similar vectors to the query vector.

        Args:
            query: Query vector
            top_k: Number of results to return

        Returns:
            List of dictionaries containing search results with id, score, and payload
        """
        conn = self._get_connection()

        # Serialize query as JSON
        query_json = json.dumps(query)

        # Execute KNN search
        cursor = conn.execute(f"""
            SELECT
                rowid,
                distance,
                row_id,
                track,
                artist,
                genre,
                seeds,
                text
            FROM {self.table_name}
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
        """, (query_json, top_k))

        # Format results
        results = []
        for row in cursor:
            results.append({
                "id": row[0],  # rowid
                "score": float(row[1]),  # distance
                "payload": {
                    "row_id": row[2],
                    "track": row[3],
                    "artist": row[4],
                    "genre": row[5],
                    "seeds": row[6],
                    "text": row[7]
                }
            })

        return results

    def teardown(self):
        """Clean up database resources."""
        if self.conn:
            try:
                # Drop the virtual table
                self.conn.execute(f"DROP TABLE IF EXISTS {self.table_name}")
                self.conn.commit()
            except Exception as e:
                print(f"Warning: Error during teardown: {e}")

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np


class MemoryStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = threading.RLock()

        self.conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
        )

        self.conn.execute("PRAGMA journal_mode=WAL")

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                timestamp INTEGER,
                content_json TEXT NOT NULL,
                retrieval_text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(request_id, ordinal)
            )
            """
        )

        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id)"
        )

        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mem_req ON memories(request_id)"
        )

        self.conn.commit()

    def request_exists(self, request_id: str) -> bool:
        with self.lock:
            row = self.conn.execute(
                "SELECT 1 FROM memories WHERE request_id=? LIMIT 1",
                (request_id,),
            ).fetchone()

        return row is not None

    def add_many(self, records: List[Dict]):
        with self.lock:
            cursor = self.conn.cursor()

            try:
                cursor.execute("BEGIN")

                for r in records:
                    vector = np.asarray(
                        r["embedding"],
                        dtype=np.float32,
                    )

                    cursor.execute(
                        """
                        INSERT INTO memories (
                            id, request_id, ordinal, user_id,
                            session_id, role, timestamp,
                            content_json, retrieval_text,
                            embedding, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            r["id"],
                            r["request_id"],
                            r["ordinal"],
                            r["user_id"],
                            r["session_id"],
                            r["role"],
                            r["timestamp"],
                            json.dumps(
                                r["content"],
                                ensure_ascii=False,
                            ),
                            r["retrieval_text"],
                            vector.tobytes(),
                            r["created_at"],
                        ),
                    )

                self.conn.commit()

            except Exception:
                self.conn.rollback()
                raise

    def get_user_memories(self, user_id: str) -> List[Dict]:
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT id, content_json, retrieval_text,
                       embedding, created_at
                FROM memories
                WHERE user_id=?
                ORDER BY rowid ASC
                """,
                (user_id,),
            ).fetchall()

        results = []

        for row in rows:
            results.append(
                {
                    "id": row[0],
                    "content": json.loads(row[1]),
                    "retrieval_text": row[2],
                    "embedding": np.frombuffer(
                        row[3],
                        dtype=np.float32,
                    ).copy(),
                    "created_at": row[4],
                }
            )

        return results

    @staticmethod
    def now_iso():
        return datetime.now(timezone.utc).isoformat()
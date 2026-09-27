from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SpoolItem:
    id: str
    path: Path
    metadata: dict[str, Any]
    attempts: int
    created_at: float


class EncryptedSpool:
    def __init__(
        self, root: str | Path, database: str | Path, max_bytes: int, max_items: int, max_age_s: int,
        encrypt: bool = True, key: bytes | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.max_bytes, self.max_items, self.max_age_s = int(max_bytes), int(max_items), int(max_age_s)
        if min(self.max_bytes, self.max_items, self.max_age_s) <= 0:
            raise ValueError("Spool capacity and retention limits must be positive")
        self.encrypt, self.key = bool(encrypt), key
        if self.encrypt and not self.key:
            raise ValueError("An encryption key is required when spool encryption is enabled")
        if self.encrypt and len(self.key) not in {16, 24, 32}:
            raise ValueError("AES-GCM key must contain 16, 24, or 32 bytes")
        self.lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS items (id TEXT PRIMARY KEY, path TEXT NOT NULL, metadata TEXT NOT NULL, "
                "status TEXT NOT NULL, attempts INTEGER NOT NULL, created REAL NOT NULL, next_attempt REAL NOT NULL, lease_until REAL, bytes INTEGER NOT NULL)"
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(items)").fetchall()}
            if "bytes" not in columns:
                connection.execute("ALTER TABLE items ADD COLUMN bytes INTEGER NOT NULL DEFAULT 0")
                for row in connection.execute("SELECT id, path FROM items").fetchall():
                    path = Path(row["path"])
                    connection.execute("UPDATE items SET bytes=? WHERE id=?", (path.stat().st_size if path.is_file() else 0, row["id"]))
            connection.execute("CREATE INDEX IF NOT EXISTS idx_items_status_next ON items(status, next_attempt)")
            connection.execute("CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL)")

    def enqueue(self, source_path: str | Path, metadata: dict[str, Any]) -> SpoolItem:
        self.expire()
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        payload = source.read_bytes()
        stored = self._encrypt(payload) if self.encrypt else payload
        item_id = str(uuid.uuid4())
        destination = self.root / f"{item_id}{'.enc' if self.encrypt else source.suffix}"
        temporary = destination.with_suffix(destination.suffix + ".partial")
        created = time.time()
        with self.lock:
            self._enforce_capacity(len(stored))
            try:
                with temporary.open("wb") as handle:
                    handle.write(stored)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.replace(destination)
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    connection.execute(
                        "INSERT INTO items VALUES (?, ?, ?, 'queued', 0, ?, ?, NULL, ?)",
                        (item_id, str(destination), json.dumps(metadata, separators=(",", ":"), default=str), created, created, len(stored)),
                    )
                    connection.commit()
            except Exception:
                temporary.unlink(missing_ok=True)
                destination.unlink(missing_ok=True)
                raise
        return SpoolItem(item_id, destination, metadata, 0, created)

    def lease(self, lease_s: int = 300) -> SpoolItem | None:
        now = time.time()
        with self.lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM items WHERE (status='queued' AND next_attempt<=?) OR (status='leased' AND lease_until<?) "
                "ORDER BY created LIMIT 1", (now, now),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            connection.execute("UPDATE items SET status='leased', attempts=attempts+1, lease_until=? WHERE id=?", (now + lease_s, row["id"]))
            connection.commit()
            return SpoolItem(row["id"], Path(row["path"]), json.loads(row["metadata"]), int(row["attempts"]) + 1, float(row["created"]))

    def materialize(self, item: SpoolItem, destination: str | Path) -> Path:
        payload = item.path.read_bytes()
        output = Path(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(self._decrypt(payload) if self.encrypt else payload)
        return output

    def acknowledge(self, item_id: str) -> None:
        with self.lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT path FROM items WHERE id=?", (item_id,)).fetchone()
            connection.execute("DELETE FROM items WHERE id=?", (item_id,))
            connection.commit()
        if row:
            Path(row["path"]).unlink(missing_ok=True)

    def retry(self, item_id: str, delay_s: float) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE items SET status='queued', next_attempt=?, lease_until=NULL WHERE id=?", (time.time() + max(0.0, delay_s), item_id))

    def dead_letter(self, item_id: str) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE items SET status='dead', lease_until=NULL WHERE id=?", (item_id,))
            self._increment(connection, "dead_total", 1)

    def expire(self) -> int:
        cutoff = time.time() - self.max_age_s
        with self.lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute("SELECT id, path FROM items WHERE created<? AND status!='dead'", (cutoff,)).fetchall()
            connection.executemany("DELETE FROM items WHERE id=?", [(row["id"],) for row in rows])
            self._increment(connection, "expired_total", len(rows))
            connection.commit()
        for row in rows:
            Path(row["path"]).unlink(missing_ok=True)
        return len(rows)

    def stats(self) -> dict[str, int | float | None]:
        now = time.time()
        with self._connect() as connection:
            rows = connection.execute("SELECT status, COUNT(*) AS count, COALESCE(SUM(bytes), 0) AS bytes FROM items GROUP BY status").fetchall()
            oldest = connection.execute("SELECT MIN(created) AS created FROM items WHERE status!='dead'").fetchone()["created"]
            counters = {row["name"]: int(row["value"]) for row in connection.execute("SELECT name, value FROM counters").fetchall()}
        result: dict[str, int | float | None] = {"items": 0, "bytes": 0, "oldest_age_s": None, **counters}
        for row in rows:
            status = str(row["status"])
            result[status] = int(row["count"])
            result[f"{status}_bytes"] = int(row["bytes"])
            result["items"] = int(result["items"]) + int(row["count"])
            result["bytes"] = int(result["bytes"]) + int(row["bytes"])
        result["oldest_age_s"] = max(0.0, now - float(oldest)) if oldest is not None else None
        return result

    def _enforce_capacity(self, incoming: int) -> None:
        stats = self.stats()
        if int(stats["items"]) >= self.max_items or int(stats["bytes"]) + incoming > self.max_bytes:
            raise RuntimeError("Edge spool capacity guardrail reached; ingestion is paused to prevent silent data loss")

    @staticmethod
    def _increment(connection: sqlite3.Connection, name: str, amount: int) -> None:
        if amount <= 0:
            return
        connection.execute(
            "INSERT INTO counters(name, value) VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET value=value+excluded.value", (name, amount),
        )

    def _encrypt(self, payload: bytes) -> bytes:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        nonce = os.urandom(12)
        return nonce + AESGCM(self.key).encrypt(nonce, payload, None)

    def _decrypt(self, payload: bytes) -> bytes:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        return AESGCM(self.key).decrypt(payload[:12], payload[12:], None)


def parse_spool_key(value: str) -> bytes:
    try:
        raw = bytes.fromhex(value)
    except ValueError:
        raw = hashlib.sha256(value.encode()).digest()
    return raw if len(raw) in {16, 24, 32} else hashlib.sha256(raw).digest()

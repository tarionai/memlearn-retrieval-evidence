"""In-memory KV store adapter — deterministic test port for KVStoreAdapter Protocol.

Backed by a plain dict[str, bytes]. No external dependencies. No network I/O.
Satisfies KVStoreAdapter (ports.py) structurally via duck-typing — no base class.
"""
from __future__ import annotations

from typing import Iterator, Optional, Tuple


class InMemoryKVStore:
    """Volatile, in-process key-value store.

    Thread-safety is intentionally not guaranteed — test usage is single-threaded.
    scan_prefix snapshots the dict at call time so iteration is safe against
    mutations that arrive during a scan.
    """

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def set(self, key: str, value: bytes) -> None:
        self._store[key] = value

    def get(self, key: str) -> Optional[bytes]:
        return self._store.get(key)

    def delete(self, key: str) -> None:
        # No-op on missing key — matches ForgetPolicy.Hard contract without raising.
        self._store.pop(key, None)

    def scan_prefix(self, prefix: str) -> Iterator[Tuple[str, bytes]]:
        # Snapshot at call time; prefix="" yields all entries.
        snapshot = list(self._store.items())
        for k, v in snapshot:
            if k.startswith(prefix):
                yield k, v

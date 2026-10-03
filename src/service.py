"""One shared Athena instance for the API and the MCP server.

On first use it connects to the configured store and, if the default strategy
has no chunks yet (always true for the in-memory store), indexes the corpus.
"""

from __future__ import annotations

import os
import threading

from .athena import Athena
from .config import settings

_lock = threading.Lock()
_instance: Athena | None = None


def get_athena() -> Athena:
    global _instance
    with _lock:
        if _instance is None:
            athena = Athena()
            auto = os.getenv("ATHENA_AUTO_INGEST", "true").lower() in ("1", "true", "yes")
            if auto and athena.store.count(settings.default_strategy) == 0:
                athena.ingest()
            _instance = athena
        return _instance


def set_athena(athena: Athena | None) -> None:
    """Replace the shared instance (used by tests)."""
    global _instance
    with _lock:
        _instance = athena

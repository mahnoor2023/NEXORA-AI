from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Any

MAX_CACHED_TRAINING_RESULTS = 20
_results: OrderedDict[str, dict[str, Any]] = OrderedDict()
_lock = threading.Lock()


def dataset_fingerprint(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def remember_training_result(raw: bytes, result: dict[str, Any]) -> None:
    fingerprint = dataset_fingerprint(raw)
    with _lock:
        _results[fingerprint] = result
        _results.move_to_end(fingerprint)
        while len(_results) > MAX_CACHED_TRAINING_RESULTS:
            _results.popitem(last=False)


def retrieve_training_result(raw: bytes) -> dict[str, Any] | None:
    fingerprint = dataset_fingerprint(raw)
    with _lock:
        result = _results.get(fingerprint)
        if result is not None:
            _results.move_to_end(fingerprint)
        return result


def forget_user_training_results(user_id: str) -> None:
    """Remove the deleted account's transient model summaries from process memory."""
    with _lock:
        for fingerprint in [key for key, result in _results.items() if result.get("owner_id") == user_id]:
            _results.pop(fingerprint, None)

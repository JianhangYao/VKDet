"""Process-level in-memory cache: written by ASKD / PAPL preprocessing,
read when the model / dataset are built.

Purpose: keep ASKD/PAPL intermediate outputs (imgembed / imgproposal /
prototypes) in memory instead of disk pkl files, so that a single
`python tools/train.py <config>` command covers preprocessing + training.

Agreed keys (a single training run uses one dataset, so keys are fixed):
    'askd_imgembed'    -> CLIP image embeddings (list/dict of np.float16)
    'askd_imgproposal' -> enhanced proposals (list of ndarray[N, 5])
    'papl_prototypes'  -> PAPL clustering outputs (Stage 2)

On read, model/dataset code checks the cache first and falls back to disk
pkl when empty (keeps the legacy flow working).
"""
from typing import Any, Optional

_CACHE = {}


def set(key: str, obj: Any) -> None:
    _CACHE[key] = obj


def get(key: str, default: Any = None) -> Any:
    return _CACHE.get(key, default)


def has(key: str) -> bool:
    return key in _CACHE


def clear() -> None:
    _CACHE.clear()

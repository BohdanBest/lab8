"""Core package for Lab 8: Transfer Learning in Keras (Dogs vs. Cats).

Configures local writable caches (KERAS_HOME, MPLCONFIGDIR) and automatically selects
the PyTorch MPS GPU backend for Keras 3 on Apple Silicon when available.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from pathlib import Path

if "google.colab" not in sys.modules:
    _root = Path(__file__).resolve().parent.parent
    _keras_cache = _root / "data" / "keras_cache"
    _mpl_cache = _root / "data" / "mpl_cache"
    _keras_cache.mkdir(parents=True, exist_ok=True)
    _mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("KERAS_HOME", str(_keras_cache))
    os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

    # Enable Apple Silicon GPU (Metal MPS) via Keras 3 multi-backend when torch is installed
    if platform.system() == "Darwin" and importlib.util.find_spec("torch") is not None:
        os.environ.setdefault("KERAS_BACKEND", "torch")

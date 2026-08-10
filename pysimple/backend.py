"""Array-backend selection kept deliberately small and explicit."""

import numpy as np


def array_module(name="numpy"):
    """Return NumPy or lazily import CuPy for the selected backend."""
    if name == "numpy":
        return np
    if name == "cupy":
        try:
            import cupy as cp
        except ImportError as error:
            raise ImportError("The CuPy backend requires PySIMPLE[gpu].") from error
        return cp
    raise ValueError("backend must be 'numpy' or 'cupy'")


def to_numpy(value, xp):
    """Return an array on the host without importing CuPy for NumPy runs."""
    return value if xp is np else xp.asnumpy(value)

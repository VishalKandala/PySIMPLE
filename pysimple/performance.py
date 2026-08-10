"""Repeatable backend benchmark helpers with no solver-specific timing magic."""

from statistics import median
from time import perf_counter


def benchmark(case, backend="numpy", repeats=3):
    """Return median wall time and convergence information for complete solves."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    from .solver import SimpleSolver
    timings, final = [], None
    for _ in range(repeats):
        solver = SimpleSolver(case, backend=backend)
        started = perf_counter()
        final = solver.solve()
        timings.append(perf_counter() - started)
    return {
        "backend": backend,
        "repeats": repeats,
        "samples_seconds": timings,
        "median_seconds": median(timings),
        "iterations": final.iterations,
        "median_iteration_milliseconds": 1.0e3 * median(timings) / final.iterations,
        "converged": final.converged,
        "final_continuity": float(final.continuity_history[-1]),
    }

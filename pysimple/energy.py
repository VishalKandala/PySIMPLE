"""GPU-portable passive energy equation on SIMPLE's pressure grid."""

from dataclasses import dataclass

import numpy as np

from .backend import to_numpy
from .config import ThermalBoundaryKind


@dataclass(frozen=True)
class ThermalResult:
    temperature: np.ndarray
    residual_history: np.ndarray
    iterations: int
    converged: bool


def _apply_boundaries(temperature, problem):
    for side, index in (("left", 0), ("right", -1), ("bottom", 0), ("top", -1)):
        condition = problem.boundaries[side]
        source = 1 if index == 0 else -2
        if side in ("left", "right"):
            temperature[index, :] = (2.0 * condition.temperature - temperature[source, :]
                                     if condition.kind is ThermalBoundaryKind.FIXED_TEMPERATURE else temperature[source, :])
        else:
            temperature[:, index] = (2.0 * condition.temperature - temperature[:, source]
                                     if condition.kind is ThermalBoundaryKind.FIXED_TEMPERATURE else temperature[:, source])


def solve_energy(flow_result, grid, fluid, problem, xp):
    """Solve steady advection-diffusion for temperature on a supplied velocity field."""
    temperature = xp.zeros(grid.pressure_shape, dtype=float)
    u, v = xp.asarray(flow_result.u), xp.asarray(flow_result.v)
    rho, k, dx, dy = fluid.density, problem.conductivity, grid.dx, grid.dy
    de, dn = k * dy / dx, k * dx / dy
    history = []
    for iteration in range(1, problem.max_iterations + 1):
        _apply_boundaries(temperature, problem)
        fe, fw = rho * u[1:, 1:-1] * dy, rho * u[:-1, 1:-1] * dy
        fn, fs = rho * v[1:-1, 1:] * dx, rho * v[1:-1, :-1] * dx
        ae, aw = de + xp.maximum(-fe, 0.0), de + xp.maximum(fw, 0.0)
        an, ass = dn + xp.maximum(-fn, 0.0), dn + xp.maximum(fs, 0.0)
        ap = ae + aw + an + ass + (fe - fw) + (fn - fs)
        for _ in range(problem.sweeps_per_iteration):
            _apply_boundaries(temperature, problem)
            previous = temperature[1:-1, 1:-1].copy()
            temperature[1:-1, 1:-1] = (
                ae * temperature[2:, 1:-1] + aw * temperature[:-2, 1:-1]
                + an * temperature[1:-1, 2:] + ass * temperature[1:-1, :-2]
            ) / ap
        residual = xp.max(xp.abs(temperature[1:-1, 1:-1] - previous))
        value = float(residual.get()) if xp is not np else float(residual)
        history.append(value)
        if value <= problem.tolerance:
            _apply_boundaries(temperature, problem)
            return ThermalResult(to_numpy(temperature, xp), np.asarray(history), iteration, True)
    _apply_boundaries(temperature, problem)
    return ThermalResult(to_numpy(temperature, xp), np.asarray(history), problem.max_iterations, False)

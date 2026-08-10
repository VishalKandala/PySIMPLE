"""Optional output and CFD quantities derived from a completed SIMPLE run."""

from pathlib import Path

import numpy as np


def cell_center_velocities(result):
    """Interpolate staggered face velocities onto physical pressure cells."""
    u = 0.5 * (result.u[:-1, 1:-1] + result.u[1:, 1:-1])
    v = 0.5 * (result.v[1:-1, :-1] + result.v[1:-1, 1:])
    return u, v


def channel_metrics(result, grid, fluid):
    """Return bulk velocity, wall shear and Fanning friction factor estimates."""
    u, _ = cell_center_velocities(result)
    bulk_velocity = float(np.mean(u[-1]))
    wall_shear = float(fluid.viscosity * (u[-1, 0] / (0.5 * grid.dy)))
    fanning = wall_shear / (0.5 * fluid.density * bulk_velocity ** 2) if bulk_velocity else np.nan
    return {"bulk_velocity": bulk_velocity, "wall_shear": wall_shear, "fanning_friction_factor": fanning}


def save_result(path, result, case):
    """Write a self-describing portable NumPy archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, u=result.u, v=result.v, pressure=result.pressure,
        continuity_history=result.continuity_history, momentum_history=result.momentum_history,
        iterations=result.iterations, converged=result.converged, case_name=case.name,
    )


def plot_result(path, result, grid):
    """Save pressure, velocity magnitude, and streamlines; requires matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("Plotting requires PySIMPLE[viz].") from error
    u, v = cell_center_velocities(result)
    x, y = grid.pressure_centres()
    speed = np.hypot(u, v)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for axis, data, title in ((axes[0], result.pressure[1:-1, 1:-1], "Pressure"), (axes[1], speed, "Speed")):
        contour = axis.contourf(x, y, data, levels=30)
        figure.colorbar(contour, ax=axis)
        axis.set_title(title); axis.set_aspect("equal")
    axes[2].streamplot(x.T, y.T, u.T, v.T, density=1.5)
    axes[2].set_title("Streamlines"); axes[2].set_aspect("equal")
    figure.savefig(path, dpi=160)
    plt.close(figure)

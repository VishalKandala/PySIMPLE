"""Structured geometry generators reserved for the non-Cartesian solver phase."""

from dataclasses import dataclass

import numpy as np


def stretched_faces(length, cells, ratio=1.0):
    """Return monotonic 1-D faces with geometric spacing and exact endpoints."""
    if length <= 0.0 or cells < 2 or ratio <= 0.0:
        raise ValueError("length, cells and ratio must be positive")
    if np.isclose(ratio, 1.0):
        return np.linspace(0.0, length, cells + 1)
    first = length * (1.0 - ratio) / (1.0 - ratio ** cells)
    widths = first * ratio ** np.arange(cells)
    return np.concatenate(([0.0], np.cumsum(widths)))


@dataclass(frozen=True)
class BodyFittedStepGrid:
    """Vertex geometry for a backward-facing step with a blended upper wall.

    It deliberately contains geometry only. The current finite-volume kernels
    use orthogonal uniform metrics; this object is the validated mesh contract
    for the subsequent curvilinear operator implementation.
    """

    vertices: np.ndarray
    step_x: float
    step_height: float

    @property
    def nx(self):
        return self.vertices.shape[0] - 1

    @property
    def ny(self):
        return self.vertices.shape[1] - 1

    @property
    def cell_areas(self):
        lower_left = self.vertices[:-1, :-1]
        lower_right = self.vertices[1:, :-1]
        upper_right = self.vertices[1:, 1:]
        upper_left = self.vertices[:-1, 1:]
        return 0.5 * np.abs(
            lower_left[..., 0] * lower_right[..., 1] - lower_left[..., 1] * lower_right[..., 0]
            + lower_right[..., 0] * upper_right[..., 1] - lower_right[..., 1] * upper_right[..., 0]
            + upper_right[..., 0] * upper_left[..., 1] - upper_right[..., 1] * upper_left[..., 0]
            + upper_left[..., 0] * lower_left[..., 1] - upper_left[..., 1] * lower_left[..., 0]
        )


def backward_facing_step_grid(nx=80, ny=32, length=8.0, inlet_height=1.0, step_x=2.0, step_height=0.5):
    """Build a positive-area body-fitted mesh for a canonical step geometry."""
    if not (nx >= 4 and ny >= 4 and length > step_x > 0.0 and inlet_height > 0.0 and step_height > 0.0):
        raise ValueError("invalid backward-facing-step dimensions")
    x = np.linspace(0.0, length, nx + 1)
    lower = np.where(x < step_x, 0.0, -step_height)
    upper = np.full_like(x, inlet_height)
    eta = np.linspace(0.0, 1.0, ny + 1)
    vertices = np.empty((nx + 1, ny + 1, 2))
    vertices[..., 0] = x[:, None]
    vertices[..., 1] = lower[:, None] + eta[None, :] * (upper - lower)[:, None]
    grid = BodyFittedStepGrid(vertices, step_x, step_height)
    if np.any(grid.cell_areas <= 0.0):
        raise ValueError("generated step mesh has non-positive cells")
    return grid

"""Geometry and array layouts for a two-dimensional staggered grid."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .config import GridSpec


@dataclass(frozen=True)
class StaggeredGrid:
    """Uniform MAC grid with one ghost layer for pressure and face velocities.

    Axis zero is x and axis one is y.  Pressure has shape ``(nx+2, ny+2)``;
    u has shape ``(nx+1, ny+2)`` and v has shape ``(nx+2, ny+1)``.
    """

    spec: GridSpec

    @property
    def dx(self) -> float:
        return self.spec.dx

    @property
    def dy(self) -> float:
        return self.spec.dy

    @property
    def pressure_shape(self) -> Tuple[int, int]:
        return self.spec.nx + 2, self.spec.ny + 2

    @property
    def u_shape(self) -> Tuple[int, int]:
        return self.spec.nx + 1, self.spec.ny + 2

    @property
    def v_shape(self) -> Tuple[int, int]:
        return self.spec.nx + 2, self.spec.ny + 1

    @property
    def pressure_interior(self) -> Tuple[slice, slice]:
        return slice(1, -1), slice(1, -1)

    def pressure_centres(self) -> Tuple[np.ndarray, np.ndarray]:
        x = (np.arange(self.spec.nx) + 0.5) * self.dx
        y = (np.arange(self.spec.ny) + 0.5) * self.dy
        return np.meshgrid(x, y, indexing="ij")

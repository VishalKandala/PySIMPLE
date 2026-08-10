"""Mutable staggered solution fields owned by one SIMPLE solve."""

from dataclasses import dataclass


@dataclass
class FlowState:
    u: object
    v: object
    pressure: object
    pressure_correction: object

    @classmethod
    def zeros(cls, grid, xp):
        return cls(
            u=xp.zeros(grid.u_shape, dtype=float),
            v=xp.zeros(grid.v_shape, dtype=float),
            pressure=xp.zeros(grid.pressure_shape, dtype=float),
            pressure_correction=xp.zeros(grid.pressure_shape, dtype=float),
        )

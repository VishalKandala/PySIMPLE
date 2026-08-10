"""Boundary operators shared by NumPy and CuPy SIMPLE runs."""

from .config import BoundaryKind


def apply_velocity_boundaries(state, grid, case) -> None:
    """Apply the constant face-velocity conditions in-place.

    Pressure-outlet velocity values are extrapolated after each correction;
    wall and inlet values are imposed directly on the appropriate faces.
    """
    b = case.boundaries
    u, v = state.u, state.v

    if b["left"].kind is BoundaryKind.PERIODIC or b["right"].kind is BoundaryKind.PERIODIC:
        if b["left"].kind is not BoundaryKind.PERIODIC or b["right"].kind is not BoundaryKind.PERIODIC:
            raise ValueError("periodic boundaries must be paired")
        u[0, :] = u[-2, :]
        u[-1, :] = u[1, :]
        v[0, :] = v[-2, :]
        v[-1, :] = v[1, :]
    else:
        for side, index in (("left", 0), ("right", -1)):
            condition = b[side]
            if condition.kind is BoundaryKind.PRESSURE_OUTLET:
                u[index, :] = u[1 if index == 0 else -2, :]
                v[index, :] = v[1 if index == 0 else -2, :]
            else:
                u[index, :] = condition.u
                v[index, :] = 2.0 * condition.v - v[1 if index == 0 else -2, :]

    # u is normal to left/right walls, while v is tangential there.  Tangential
    # values are stored at half-cell locations, so a wall value is imposed by
    # reflection into the ghost layer (2*u_wall - u_adjacent).
    # v is normal to bottom/top walls, while u is tangential there.
    for side, index in (("bottom", 0), ("top", -1)):
        condition = b[side]
        if condition.kind is BoundaryKind.PRESSURE_OUTLET:
            u[:, index] = u[:, 1 if index == 0 else -2]
            v[:, index] = v[:, 1 if index == 0 else -2]
        else:
            u[:, index] = 2.0 * condition.u - u[:, 1 if index == 0 else -2]
            v[:, index] = condition.v


def apply_pressure_boundaries(pressure, case) -> None:
    """Apply pressure outlet values and zero-gradient values elsewhere."""
    b = case.boundaries
    if b["left"].kind is BoundaryKind.PERIODIC:
        pressure[0, :] = pressure[-2, :]
        pressure[-1, :] = pressure[1, :]
    else:
        for side, index in (("left", 0), ("right", -1)):
            condition = b[side]
            source = 1 if index == 0 else -2
            pressure[index, :] = condition.pressure if condition.kind is BoundaryKind.PRESSURE_OUTLET else pressure[source, :]
    
    for side, index in (("bottom", 0), ("top", -1)):
        condition = b[side]
        source = 1 if index == 0 else -2
        pressure[:, index] = condition.pressure if condition.kind is BoundaryKind.PRESSURE_OUTLET else pressure[:, source]


def apply_pressure_correction_boundaries(correction, case) -> None:
    """Set p' to zero at pressure outlets and use zero normal gradient otherwise."""
    b = case.boundaries
    if b["left"].kind is BoundaryKind.PERIODIC:
        correction[0, :] = correction[-2, :]
        correction[-1, :] = correction[1, :]
    else:
        for side, index in (("left", 0), ("right", -1)):
            condition = b[side]
            source = 1 if index == 0 else -2
            correction[index, :] = 0.0 if condition.kind is BoundaryKind.PRESSURE_OUTLET else correction[source, :]
    for side, index in (("bottom", 0), ("top", -1)):
        condition = b[side]
        source = 1 if index == 0 else -2
        correction[:, index] = 0.0 if condition.kind is BoundaryKind.PRESSURE_OUTLET else correction[:, source]

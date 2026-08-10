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


def enforce_outlet_mass_flux(state, grid, case, xp) -> None:
    """Scale a pressure-outlet profile to match a prescribed inlet flux.

    SIMPLE's pressure correction enforces local continuity in the domain.  A
    velocity-inlet/pressure-outlet pair additionally needs a consistent outlet
    flux while the pressure field is settling.  This conservative correction is
    restricted to that boundary pairing and leaves periodic/cavity cases alone.
    """
    left = case.boundaries["left"]
    right = case.boundaries["right"]
    if left.kind is not BoundaryKind.VELOCITY_INLET or right.kind is not BoundaryKind.PRESSURE_OUTLET:
        return
    inlet_flux = xp.sum(state.u[0, 1:-1])
    outlet_profile = state.u[-2, 1:-1]
    outlet_flux = xp.sum(outlet_profile)
    valid_flux = xp.abs(outlet_flux) > 1.0e-30
    scaled_profile = outlet_profile * inlet_flux / xp.where(valid_flux, outlet_flux, 1.0)
    state.u[-1, 1:-1] = xp.where(valid_flux, scaled_profile, state.u[0, 1:-1])

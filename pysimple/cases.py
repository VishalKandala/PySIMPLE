"""Named baseline cases used by examples and verification tests."""

from .config import (
    BoundaryCondition, BoundaryKind, FlowCase, Fluid, GridSpec, SimpleControls,
    ThermalBoundaryCondition, ThermalBoundaryKind, ThermalProblem,
)


def lid_driven_cavity(nx=32, ny=32, reynolds=100.0, lid_velocity=1.0) -> FlowCase:
    """Return a unit-square lid-driven cavity case.

    The viscosity is selected so that ``Re = rho * U_lid * L / mu`` with
    density one.  This is the first supported physical validation case.
    """
    if reynolds <= 0.0 or lid_velocity <= 0.0:
        raise ValueError("reynolds and lid_velocity must be positive")
    return FlowCase(
        name="lid-driven-cavity",
        grid=GridSpec(nx=nx, ny=ny),
        fluid=Fluid(density=1.0, viscosity=lid_velocity / reynolds),
        controls=SimpleControls(),
        boundaries={
            "left": BoundaryCondition(BoundaryKind.WALL),
            "right": BoundaryCondition(BoundaryKind.WALL),
            "bottom": BoundaryCondition(BoundaryKind.WALL),
            "top": BoundaryCondition(BoundaryKind.WALL, u=lid_velocity),
        },
    )


def developing_channel(nx=80, ny=24, length=8.0, height=1.0, reynolds=100.0, inlet_velocity=1.0) -> FlowCase:
    """Return an isothermal channel with a uniform inlet and fixed outlet p."""
    if reynolds <= 0.0 or inlet_velocity <= 0.0:
        raise ValueError("reynolds and inlet_velocity must be positive")
    viscosity = inlet_velocity * (2.0 * height) / reynolds
    return FlowCase(
        name="developing-channel",
        grid=GridSpec(nx=nx, ny=ny, length=length, height=height),
        fluid=Fluid(density=1.0, viscosity=viscosity),
        controls=SimpleControls(max_iterations=4_000, continuity_tolerance=1.0e-7),
        boundaries={
            "left": BoundaryCondition(BoundaryKind.VELOCITY_INLET, u=inlet_velocity),
            "right": BoundaryCondition(BoundaryKind.PRESSURE_OUTLET, pressure=0.0),
            "bottom": BoundaryCondition(BoundaryKind.WALL),
            "top": BoundaryCondition(BoundaryKind.WALL),
        },
    )


def fully_developed_channel(nx=16, ny=32, height=1.0, viscosity=0.01, body_force_x=0.08) -> FlowCase:
    """Return a periodic pressure-gradient-equivalent Poiseuille benchmark.

    ``body_force_x`` is acceleration. The analytic velocity is
    ``rho * body_force_x * y * (height - y) / (2 * viscosity)``.
    """
    return FlowCase(
        name="fully-developed-channel",
        grid=GridSpec(nx=nx, ny=ny, length=height, height=height),
        fluid=Fluid(density=1.0, viscosity=viscosity),
        controls=SimpleControls(max_iterations=3_000, continuity_tolerance=1.0e-10),
        boundaries={
            "left": BoundaryCondition(BoundaryKind.PERIODIC),
            "right": BoundaryCondition(BoundaryKind.PERIODIC),
            "bottom": BoundaryCondition(BoundaryKind.WALL),
            "top": BoundaryCondition(BoundaryKind.WALL),
        },
        body_force_x=body_force_x,
    )


def heated_channel_problem(inlet_temperature=300.0, wall_temperature=350.0, conductivity=0.6, specific_heat=4_180.0):
    """Return a fixed-wall-temperature passive energy problem for a channel."""
    return ThermalProblem(
        conductivity=conductivity,
        specific_heat=specific_heat,
        boundaries={
            "left": ThermalBoundaryCondition(ThermalBoundaryKind.FIXED_TEMPERATURE, inlet_temperature),
            "right": ThermalBoundaryCondition(ThermalBoundaryKind.ZERO_GRADIENT),
            "bottom": ThermalBoundaryCondition(ThermalBoundaryKind.FIXED_TEMPERATURE, wall_temperature),
            "top": ThermalBoundaryCondition(ThermalBoundaryKind.FIXED_TEMPERATURE, wall_temperature),
        },
    )


def available_cases():
    """Return the stable names accepted by the public CLI and API."""
    return ("cavity", "channel", "poiseuille")


def make_case(name, nx=None, ny=None, reynolds=100.0, lid_velocity=1.0):
    """Build a named baseline case with approachable common overrides."""
    if name == "cavity":
        return lid_driven_cavity(nx=nx or 32, ny=ny or 32, reynolds=reynolds, lid_velocity=lid_velocity)
    if name == "channel":
        return developing_channel(nx=nx or 80, ny=ny or 24, reynolds=reynolds, inlet_velocity=lid_velocity)
    if name == "poiseuille":
        return fully_developed_channel(nx=nx or 16, ny=ny or 32)
    raise ValueError("unknown case {!r}; choose from {}".format(name, ", ".join(available_cases())))

"""Validated, immutable configuration objects for SIMPLE cases."""

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Dict


def _positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0.0:
        raise ValueError("{} must be finite and positive; got {!r}".format(name, value))


class BoundaryKind(str, Enum):
    WALL = "wall"
    VELOCITY_INLET = "velocity_inlet"
    PRESSURE_OUTLET = "pressure_outlet"
    PERIODIC = "periodic"


class ThermalBoundaryKind(str, Enum):
    FIXED_TEMPERATURE = "fixed_temperature"
    ZERO_GRADIENT = "zero_gradient"


@dataclass(frozen=True)
class Fluid:
    """Constant-density Newtonian fluid properties in SI units."""

    density: float = 1.0
    viscosity: float = 1.0e-3

    def __post_init__(self) -> None:
        _positive("density", self.density)
        _positive("viscosity", self.viscosity)


@dataclass(frozen=True)
class GridSpec:
    """Uniform Cartesian grid described by physical cell counts."""

    nx: int
    ny: int
    length: float = 1.0
    height: float = 1.0

    def __post_init__(self) -> None:
        if self.nx < 2 or self.ny < 2:
            raise ValueError("nx and ny must both be at least 2")
        _positive("length", self.length)
        _positive("height", self.height)

    @property
    def dx(self) -> float:
        return self.length / self.nx

    @property
    def dy(self) -> float:
        return self.height / self.ny


@dataclass(frozen=True)
class BoundaryCondition:
    """A constant velocity or pressure condition on one domain side."""

    kind: BoundaryKind
    u: float = 0.0
    v: float = 0.0
    pressure: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (("u", self.u), ("v", self.v), ("pressure", self.pressure)):
            if not isfinite(value):
                raise ValueError("{} must be finite".format(name))


@dataclass(frozen=True)
class ThermalBoundaryCondition:
    kind: ThermalBoundaryKind
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.temperature):
            raise ValueError("temperature must be finite")


@dataclass(frozen=True)
class ThermalProblem:
    """Passive scalar energy model coupled one-way to a converged flow field."""

    conductivity: float
    specific_heat: float
    boundaries: Dict[str, ThermalBoundaryCondition]
    max_iterations: int = 5_000
    sweeps_per_iteration: int = 10
    tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        _positive("conductivity", self.conductivity)
        _positive("specific_heat", self.specific_heat)
        _positive("tolerance", self.tolerance)
        if self.max_iterations < 1 or self.sweeps_per_iteration < 1:
            raise ValueError("thermal iteration counts must be positive")
        if set(self.boundaries) != {"left", "right", "bottom", "top"}:
            raise ValueError("thermal boundaries must contain all four sides")


@dataclass(frozen=True)
class SimpleControls:
    """Outer SIMPLE and inner Jacobi iteration controls."""

    max_iterations: int = 2_000
    momentum_sweeps: int = 20
    pressure_sweeps: int = 80
    pressure_solver: str = "jacobi"
    multigrid_cycles: int = 4
    momentum_relaxation: float = 0.7
    pressure_relaxation: float = 0.3
    continuity_tolerance: float = 1.0e-7
    momentum_tolerance: float = 1.0e-7
    monitor_interval: int = 50

    def __post_init__(self) -> None:
        if self.max_iterations < 1 or self.momentum_sweeps < 1 or self.pressure_sweeps < 1:
            raise ValueError("iteration counts must be positive")
        if self.pressure_solver not in {"jacobi", "multigrid"}:
            raise ValueError("pressure_solver must be 'jacobi' or 'multigrid'")
        if self.multigrid_cycles < 1:
            raise ValueError("multigrid_cycles must be positive")
        if self.monitor_interval < 1:
            raise ValueError("monitor_interval must be positive")
        for name, value in (("momentum_relaxation", self.momentum_relaxation),
                            ("pressure_relaxation", self.pressure_relaxation)):
            if not 0.0 < value <= 1.0:
                raise ValueError("{} must lie in (0, 1]".format(name))
        _positive("continuity_tolerance", self.continuity_tolerance)
        _positive("momentum_tolerance", self.momentum_tolerance)


@dataclass(frozen=True)
class FlowCase:
    """Complete, side-effect-free description of a steady incompressible case."""

    name: str
    grid: GridSpec
    fluid: Fluid = field(default_factory=Fluid)
    controls: SimpleControls = field(default_factory=SimpleControls)
    boundaries: Dict[str, BoundaryCondition] = field(default_factory=dict)
    body_force_x: float = 0.0
    body_force_y: float = 0.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be blank")
        expected = {"left", "right", "bottom", "top"}
        if set(self.boundaries) != expected:
            raise ValueError("boundaries must contain exactly {}".format(sorted(expected)))
        if not isfinite(self.body_force_x) or not isfinite(self.body_force_y):
            raise ValueError("body forces must be finite")

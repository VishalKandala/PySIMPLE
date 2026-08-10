"""Small high-level API for scripts and notebooks."""

from dataclasses import dataclass, replace

from .cases import make_case
from .solver import SimpleSolver


@dataclass(frozen=True)
class Run:
    """A named case, its grid, and the completed host-resident solution."""

    case: object
    grid: object
    result: object


def run(case="cavity", *, backend="numpy", nx=None, ny=None, reynolds=100.0,
        lid_velocity=1.0, iterations=None, tolerance=None, pressure_solver=None,
        momentum_sweeps=None, pressure_sweeps=None):
    """Build and solve a baseline case in one call.

    This is the recommended notebook entry point. Advanced users may instead
    construct a ``FlowCase`` and pass it directly to ``SimpleSolver``.
    """
    definition = make_case(case, nx=nx, ny=ny, reynolds=reynolds, lid_velocity=lid_velocity)
    controls = definition.controls
    if iterations is not None:
        controls = replace(controls, max_iterations=iterations)
    if tolerance is not None:
        controls = replace(controls, continuity_tolerance=tolerance)
    if pressure_solver is not None:
        controls = replace(controls, pressure_solver=pressure_solver)
    if momentum_sweeps is not None:
        controls = replace(controls, momentum_sweeps=momentum_sweeps)
    if pressure_sweeps is not None:
        controls = replace(controls, pressure_sweeps=pressure_sweeps)
    definition = replace(definition, controls=controls)
    solver = SimpleSolver(definition, backend=backend)
    return Run(definition, solver.grid, solver.solve())

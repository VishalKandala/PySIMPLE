"""Backend-portable SIMPLE solver primitives for Cartesian grids."""

from .api import Run, run
from .cases import (
    available_cases, developing_channel, fully_developed_channel, heated_channel_problem,
    lid_driven_cavity, make_case,
)
from .config import (
    BoundaryCondition, BoundaryKind, FlowCase, Fluid, GridSpec, SimpleControls,
    ThermalBoundaryCondition, ThermalBoundaryKind, ThermalProblem,
)
from .grid import StaggeredGrid
from .solver import SimpleResult, SimpleSolver

__all__ = [
    "BoundaryCondition", "BoundaryKind", "FlowCase", "Fluid", "GridSpec", "ThermalBoundaryCondition",
    "ThermalBoundaryKind", "ThermalProblem", "SimpleControls", "StaggeredGrid", "SimpleResult", "SimpleSolver",
    "Run", "run", "available_cases", "make_case", "developing_channel", "fully_developed_channel",
    "heated_channel_problem", "lid_driven_cavity",
]

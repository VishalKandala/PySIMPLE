"""The backend-portable, steady two-dimensional SIMPLE iteration."""

from dataclasses import dataclass

import numpy as np

from .backend import array_module, to_numpy
from .boundary import (
    apply_pressure_boundaries,
    apply_pressure_correction_boundaries,
    apply_velocity_boundaries,
)
from .grid import StaggeredGrid
from .kernels import (
    continuity_residual,
    correct_fields,
    jacobi_momentum,
    jacobi_pressure_correction,
    momentum_residual,
    momentum_coefficients,
    pressure_correction_coefficients,
)
from .state import FlowState
from .energy import solve_energy
from .multigrid import solve_pressure_multigrid


@dataclass(frozen=True)
class SimpleResult:
    """Host-resident fields and convergence history from a SIMPLE run."""

    u: np.ndarray
    v: np.ndarray
    pressure: np.ndarray
    continuity_history: np.ndarray
    momentum_history: np.ndarray
    iterations: int
    converged: bool


class SimpleSolver:
    """SIMPLE solver whose numerical kernels run against NumPy or CuPy arrays.

    The initial implementation is laminar, constant-property, two-dimensional,
    and uniform-Cartesian.  Jacobi sub-iterations are deliberately selected in
    place of line TDMA because their stencil operations map directly to GPUs.
    """

    def __init__(self, case, backend="numpy"):
        self.case = case
        self.grid = StaggeredGrid(case.grid)
        self.backend_name = backend
        self.xp = array_module(backend)

    def initial_state(self):
        state = FlowState.zeros(self.grid, self.xp)
        apply_velocity_boundaries(state, self.grid, self.case)
        apply_pressure_boundaries(state.pressure, self.case)
        return state

    def solve(self, state=None):
        """Iterate to the configured continuity tolerance.

        State remains device-resident for the complete CuPy solve.  Only the
        final result and scalar residual history are transferred to the host.
        """
        xp = self.xp
        state = self.initial_state() if state is None else state
        history = []
        momentum_history = []
        controls = self.case.controls
        for iteration in range(1, controls.max_iterations + 1):
            apply_velocity_boundaries(state, self.grid, self.case)
            u_coefficients, v_coefficients = momentum_coefficients(
                state, self.grid, self.case.fluid, controls.momentum_relaxation, xp,
                self.case.body_force_x, self.case.body_force_y,
            )
            jacobi_momentum(state.u, u_coefficients, controls.momentum_sweeps)
            apply_velocity_boundaries(state, self.grid, self.case)
            jacobi_momentum(state.v, v_coefficients, controls.momentum_sweeps)
            apply_velocity_boundaries(state, self.grid, self.case)
            u_residual = momentum_residual(state.u, u_coefficients, xp)
            v_residual = momentum_residual(state.v, v_coefficients, xp)

            ae, aw, an, ass, ap, defect, du, dv = pressure_correction_coefficients(
                state, self.grid, self.case.fluid, u_coefficients[4], v_coefficients[4], xp
            )
            state.pressure_correction.fill(0.0)
            apply_correction_bc = lambda correction: apply_pressure_correction_boundaries(correction, self.case)
            if controls.pressure_solver == "jacobi":
                jacobi_pressure_correction(
                    state.pressure_correction, (ae, aw, an, ass, ap, defect), controls.pressure_sweeps,
                    apply_correction_bc,
                )
            else:
                # Multigrid operates on the same variable-coefficient stencil.
                # The pinned reference removes the closed-domain nullspace.
                ap[1, 1] = 1.0; ae[1, 1] = aw[1, 1] = an[1, 1] = ass[1, 1] = defect[1, 1] = 0.0
                solve_pressure_multigrid(
                    state.pressure_correction, (ae, aw, an, ass, ap), defect,
                    controls.multigrid_cycles, apply_correction_bc, xp,
                )
            correct_fields(state, state.pressure_correction, self.grid, controls.pressure_relaxation, du, dv)
            apply_velocity_boundaries(state, self.grid, self.case)
            apply_pressure_boundaries(state.pressure, self.case)

            residual = continuity_residual(state, self.grid, self.case.fluid.density, xp)
            residual_value = float(residual.get()) if xp is not np else float(residual)
            momentum_value = float(xp.maximum(u_residual, v_residual).get()) if xp is not np else float(max(u_residual, v_residual))
            history.append(residual_value)
            momentum_history.append(momentum_value)
            # Continuity is the governing SIMPLE convergence criterion.  The
            # momentum residual is recorded for diagnostics; each momentum
            # equation is only approximately solved during a segregated outer
            # iteration, so requiring its transient algebraic residual here
            # can incorrectly reject an otherwise converged field.
            if residual_value <= controls.continuity_tolerance:
                return self._result(state, history, momentum_history, iteration, True)
        return self._result(state, history, momentum_history, controls.max_iterations, False)

    def solve_energy(self, flow_result, problem):
        """Solve a passive thermal problem using a completed velocity field."""
        return solve_energy(flow_result, self.grid, self.case.fluid, problem, self.xp)

    def _result(self, state, history, momentum_history, iterations, converged):
        return SimpleResult(
            u=to_numpy(state.u, self.xp),
            v=to_numpy(state.v, self.xp),
            pressure=to_numpy(state.pressure, self.xp),
            continuity_history=np.asarray(history),
            momentum_history=np.asarray(momentum_history),
            iterations=iterations,
            converged=converged,
        )

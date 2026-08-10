"""Small, deterministic verification tests for the SIMPLE foundation."""

from dataclasses import replace
import unittest

import numpy as np

from pysimple import GridSpec, SimpleControls, SimpleSolver, developing_channel, fully_developed_channel, lid_driven_cavity, run
from pysimple.config import ThermalBoundaryCondition, ThermalBoundaryKind, ThermalProblem
from pysimple.solver import SimpleResult
from pysimple.geometry import backward_facing_step_grid, stretched_faces
from pysimple.verification import developing_channel_metrics


class SimpleSolverTests(unittest.TestCase):
    def test_grid_uses_staggered_shapes(self):
        case = lid_driven_cavity(nx=8, ny=6)
        grid = SimpleSolver(case).grid
        self.assertEqual(grid.pressure_shape, (10, 8))
        self.assertEqual(grid.u_shape, (9, 8))
        self.assertEqual(grid.v_shape, (10, 7))

    def test_lid_boundary_is_applied_to_initial_state(self):
        solver = SimpleSolver(lid_driven_cavity(nx=8, ny=8, lid_velocity=2.0))
        state = solver.initial_state()
        np.testing.assert_allclose(0.5 * (state.u[1:-1, -1] + state.u[1:-1, -2]), 2.0)
        np.testing.assert_allclose(state.v[:, -1], 0.0)

    def test_one_cavity_iteration_produces_finite_fields(self):
        case = lid_driven_cavity(nx=8, ny=8, reynolds=50.0)
        controls = replace(case.controls, max_iterations=1, momentum_sweeps=5, pressure_sweeps=20)
        result = SimpleSolver(replace(case, controls=controls)).solve()
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 1)
        self.assertTrue(np.isfinite(result.u).all())
        self.assertTrue(np.isfinite(result.v).all())
        self.assertTrue(np.isfinite(result.pressure).all())
        self.assertEqual(result.continuity_history.shape, (1,))

    def test_cavity_continuity_decreases(self):
        case = lid_driven_cavity(nx=12, ny=12, reynolds=100.0)
        controls = replace(
            case.controls, max_iterations=60, momentum_sweeps=12, pressure_sweeps=40,
            continuity_tolerance=1.0e-12,
        )
        result = SimpleSolver(replace(case, controls=controls)).solve()
        self.assertGreater(result.continuity_history[0], result.continuity_history[-1])

    def test_unknown_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "backend"):
            SimpleSolver(lid_driven_cavity(), backend="unsupported")

    def test_periodic_poiseuille_profile_matches_analytic_solution(self):
        case = fully_developed_channel(nx=8, ny=24, viscosity=0.01, body_force_x=0.08)
        controls = replace(case.controls, max_iterations=1_200, momentum_sweeps=20, pressure_sweeps=40,
                           continuity_tolerance=1.0e-8)
        solver = SimpleSolver(replace(case, controls=controls))
        result = solver.solve()
        y = (np.arange(case.grid.ny) + 0.5) * solver.grid.dy
        expected = case.fluid.density * case.body_force_x * y * (case.grid.height - y) / (2.0 * case.fluid.viscosity)
        profile = 0.5 * (result.u[3, 1:-1] + result.u[4, 1:-1])
        self.assertTrue(result.converged)
        np.testing.assert_allclose(profile, expected, atol=2.5e-3)

    def test_pure_conduction_matches_linear_temperature(self):
        flow_case = lid_driven_cavity(nx=4, ny=16)
        solver = SimpleSolver(flow_case)
        flow = SimpleResult(
            np.zeros(solver.grid.u_shape), np.zeros(solver.grid.v_shape), np.zeros(solver.grid.pressure_shape),
            np.asarray([]), np.asarray([]), 0, True,
        )
        thermal = ThermalProblem(
            conductivity=1.0, specific_heat=1.0,
            boundaries={
                "left": ThermalBoundaryCondition(ThermalBoundaryKind.ZERO_GRADIENT),
                "right": ThermalBoundaryCondition(ThermalBoundaryKind.ZERO_GRADIENT),
                "bottom": ThermalBoundaryCondition(ThermalBoundaryKind.FIXED_TEMPERATURE, 0.0),
                "top": ThermalBoundaryCondition(ThermalBoundaryKind.FIXED_TEMPERATURE, 1.0),
            }, tolerance=1.0e-8,
        )
        result = solver.solve_energy(flow, thermal)
        expected = (np.arange(16) + 0.5) / 16.0
        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.temperature[2, 1:-1], expected, atol=1.0e-5)

    def test_multigrid_pressure_path_produces_finite_fields(self):
        case = lid_driven_cavity(nx=8, ny=8)
        controls = replace(case.controls, max_iterations=2, pressure_solver="multigrid", multigrid_cycles=2)
        result = SimpleSolver(replace(case, controls=controls)).solve()
        self.assertTrue(np.isfinite(result.pressure).all())

    def test_stretched_and_step_geometry_are_valid(self):
        faces = stretched_faces(2.0, 10, ratio=1.1)
        self.assertEqual(faces.shape, (11,))
        self.assertTrue(np.all(np.diff(faces) > 0.0))
        mesh = backward_facing_step_grid(nx=20, ny=12)
        self.assertTrue(np.all(mesh.cell_areas > 0.0))

    def test_cupy_matches_numpy_when_available(self):
        try:
            import cupy
            if cupy.cuda.runtime.getDeviceCount() < 1:
                self.skipTest("no CUDA device is available")
        except (ImportError, RuntimeError):
            self.skipTest("CuPy/CUDA is not available on this test host")
        case = lid_driven_cavity(nx=8, ny=8)
        controls = replace(case.controls, max_iterations=4, momentum_sweeps=5, pressure_sweeps=12)
        case = replace(case, controls=controls)
        cpu = SimpleSolver(case, backend="numpy").solve()
        gpu = SimpleSolver(case, backend="cupy").solve()
        np.testing.assert_allclose(gpu.u, cpu.u, rtol=1.0e-11, atol=1.0e-12)
        np.testing.assert_allclose(gpu.v, cpu.v, rtol=1.0e-11, atol=1.0e-12)
        np.testing.assert_allclose(gpu.pressure, cpu.pressure, rtol=1.0e-11, atol=1.0e-12)

    def test_high_level_run_api_builds_and_solves_a_case(self):
        completed = run("cavity", nx=8, ny=8, iterations=1)
        self.assertEqual(completed.case.name, "lid-driven-cavity")
        self.assertEqual(completed.result.iterations, 1)
        self.assertEqual(completed.grid.spec.nx, 8)

    def test_open_channel_conserves_mass_and_reaches_poiseuille_profile(self):
        case = developing_channel(nx=48, ny=16, length=8.0, reynolds=40.0)
        controls = replace(
            case.controls, max_iterations=2_500, momentum_sweeps=20, pressure_sweeps=100,
            continuity_tolerance=5.0e-7,
        )
        solver = SimpleSolver(replace(case, controls=controls))
        result = solver.solve()
        metrics = developing_channel_metrics(result, solver.grid, case)
        self.assertTrue(result.converged)
        self.assertLess(metrics["mass_flux_relative_error"], 1.0e-12)
        self.assertLess(metrics["outlet_profile_linf_error"], 1.0e-2)
        self.assertAlmostEqual(metrics["fanning_friction_factor"], metrics["fanning_reference_24_over_re"], delta=1.0e-2)


if __name__ == "__main__":
    unittest.main()

"""Analytic-reference comparisons for reproducible solver verification."""

import numpy as np

from .postprocess import cell_center_velocities


def poiseuille_metrics(result, grid, case):
    """Compare a periodic body-force-driven channel against plane Poiseuille flow."""
    y = (np.arange(case.grid.ny) + 0.5) * grid.dy
    expected = (
        case.fluid.density * case.body_force_x * y * (case.grid.height - y)
        / (2.0 * case.fluid.viscosity)
    )
    u, v = cell_center_velocities(result)
    profile = np.mean(u, axis=0)
    error = profile - expected
    bulk_velocity = float(np.mean(profile))
    reynolds = case.fluid.density * bulk_velocity * (2.0 * case.grid.height) / case.fluid.viscosity
    wall_shear = case.fluid.density * case.body_force_x * case.grid.height / 2.0
    fanning = wall_shear / (0.5 * case.fluid.density * bulk_velocity ** 2)
    return {
        "profile_linf_error": float(np.max(np.abs(error))),
        "profile_l2_error": float(np.sqrt(np.mean(error ** 2))),
        "max_abs_v": float(np.max(np.abs(v))),
        "bulk_velocity": bulk_velocity,
        "reynolds": reynolds,
        "fanning_friction_factor": float(fanning),
        "fanning_reference_24_over_re": float(24.0 / reynolds),
    }


def developing_channel_metrics(result, grid, case):
    """Assess a velocity-inlet/pressure-outlet channel at its downstream end."""
    u, v = cell_center_velocities(result)
    bulk_velocity = float(np.mean(u[-1]))
    y = (np.arange(case.grid.ny) + 0.5) * grid.dy / case.grid.height
    reference_profile = 6.0 * bulk_velocity * y * (1.0 - y)
    profile_error = u[-1] - reference_profile
    inlet_flux = float(np.sum(result.u[0, 1:-1]) * grid.dy)
    outlet_flux = float(np.sum(result.u[-1, 1:-1]) * grid.dy)
    reynolds = case.fluid.density * bulk_velocity * (2.0 * case.grid.height) / case.fluid.viscosity
    wall_shear = float(case.fluid.viscosity * u[-1, 0] / (0.5 * grid.dy))
    fanning = wall_shear / (0.5 * case.fluid.density * bulk_velocity ** 2)
    return {
        "outlet_profile_linf_error": float(np.max(np.abs(profile_error))),
        "outlet_profile_l2_error": float(np.sqrt(np.mean(profile_error ** 2))),
        "max_abs_v": float(np.max(np.abs(v))),
        "inlet_mass_flux": inlet_flux,
        "outlet_mass_flux": outlet_flux,
        "mass_flux_relative_error": float(abs(outlet_flux - inlet_flux) / abs(inlet_flux)),
        "bulk_velocity": bulk_velocity,
        "reynolds": reynolds,
        "fanning_friction_factor": fanning,
        "fanning_reference_24_over_re": float(24.0 / reynolds),
    }

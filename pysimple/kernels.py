"""Vectorized finite-volume kernels parameterized only by an array module."""


def momentum_coefficients(state, grid, fluid, relaxation, xp, body_force_x=0.0, body_force_y=0.0):
    """Assemble first-order-upwind momentum coefficients for u and v faces."""
    rho, mu, dx, dy = fluid.density, fluid.viscosity, grid.dx, grid.dy
    u, v, pressure = state.u, state.v, state.pressure

    # u control volumes: (nx - 1, ny) internal vertical faces.
    uc = u[1:-1, 1:-1]
    fe = rho * 0.5 * (uc + u[2:, 1:-1]) * dy
    fw = rho * 0.5 * (u[:-2, 1:-1] + uc) * dy
    fn = rho * 0.5 * (v[1:-2, 1:] + v[2:-1, 1:]) * dx
    fs = rho * 0.5 * (v[1:-2, :-1] + v[2:-1, :-1]) * dx
    de, dn = mu * dy / dx, mu * dx / dy
    ue, uw = de + xp.maximum(-fe, 0.0), de + xp.maximum(fw, 0.0)
    un, us = dn + xp.maximum(-fn, 0.0), dn + xp.maximum(fs, 0.0)
    up_raw = ue + uw + un + us + (fe - fw) + (fn - fs)
    up = up_raw / relaxation
    ub = ((pressure[1:-2, 1:-1] - pressure[2:-1, 1:-1]) * dy
          + rho * body_force_x * dx * dy + (1.0 - relaxation) * up * uc)

    # v control volumes: (nx, ny - 1) internal horizontal faces.
    vc = v[1:-1, 1:-1]
    fe = rho * 0.5 * (u[1:, 1:-2] + u[1:, 2:-1]) * dy
    fw = rho * 0.5 * (u[:-1, 1:-2] + u[:-1, 2:-1]) * dy
    fn = rho * 0.5 * (vc + v[1:-1, 2:]) * dx
    fs = rho * 0.5 * (v[1:-1, :-2] + vc) * dx
    ve, vw = de + xp.maximum(-fe, 0.0), de + xp.maximum(fw, 0.0)
    vn, vs = dn + xp.maximum(-fn, 0.0), dn + xp.maximum(fs, 0.0)
    vp_raw = ve + vw + vn + vs + (fe - fw) + (fn - fs)
    vp = vp_raw / relaxation
    vb = ((pressure[1:-1, 1:-2] - pressure[1:-1, 2:-1]) * dx
          + rho * body_force_y * dx * dy + (1.0 - relaxation) * vp * vc)
    return (ue, uw, un, us, up, ub), (ve, vw, vn, vs, vp, vb)


def jacobi_momentum(field, coefficients, sweeps):
    """Perform Jacobi sweeps over one momentum field's internal faces."""
    ae, aw, an, ass, ap, source = coefficients
    for _ in range(sweeps):
        field[1:-1, 1:-1] = (
            ae * field[2:, 1:-1] + aw * field[:-2, 1:-1]
            + an * field[1:-1, 2:] + ass * field[1:-1, :-2] + source
        ) / ap


def momentum_residual(field, coefficients, xp):
    """Return the normalized algebraic residual of one momentum system."""
    ae, aw, an, ass, ap, source = coefficients
    defect = ap * field[1:-1, 1:-1] - (
        ae * field[2:, 1:-1] + aw * field[:-2, 1:-1]
        + an * field[1:-1, 2:] + ass * field[1:-1, :-2] + source
    )
    scale = xp.maximum(xp.max(xp.abs(ap * field[1:-1, 1:-1])), 1.0e-30)
    return xp.max(xp.abs(defect)) / scale


def pressure_correction_coefficients(state, grid, fluid, u_ap, v_ap, xp):
    """Assemble the p' equation and continuity defect on pressure cells."""
    nx, ny = grid.spec.nx, grid.spec.ny
    shape = grid.pressure_shape
    ae = xp.zeros(shape); aw = xp.zeros(shape); an = xp.zeros(shape); ass = xp.zeros(shape)
    rho, dx, dy = fluid.density, grid.dx, grid.dy
    du = dy / u_ap
    dv = dx / v_ap
    ae[1:-1, 1:-1] = 0.0
    aw[1:-1, 1:-1] = 0.0
    an[1:-1, 1:-1] = 0.0
    ass[1:-1, 1:-1] = 0.0
    ae[1:nx, 1:-1] = rho * dy * du
    aw[2:nx + 1, 1:-1] = rho * dy * du
    an[1:-1, 1:ny] = rho * dx * dv
    ass[1:-1, 2:ny + 1] = rho * dx * dv
    ap = ae + aw + an + ass
    defect = xp.zeros(shape)
    defect[1:-1, 1:-1] = rho * (
        (state.u[:-1, 1:-1] - state.u[1:, 1:-1]) * dy
        + (state.v[1:-1, :-1] - state.v[1:-1, 1:]) * dx
    )
    return ae, aw, an, ass, ap, defect, du, dv


def jacobi_pressure_correction(correction, coefficients, sweeps, apply_boundaries) -> None:
    """Solve p' with Jacobi iteration, pinning one value for closed domains."""
    ae, aw, an, ass, ap, defect = coefficients
    # A closed incompressible domain has a pressure nullspace.  Pinning p'[1,1]
    # supplies the needed reference while retaining zero-gradient wall values.
    ap[1, 1] = 1.0
    ae[1, 1] = aw[1, 1] = an[1, 1] = ass[1, 1] = defect[1, 1] = 0.0
    for _ in range(sweeps):
        apply_boundaries(correction)
        correction[1:-1, 1:-1] = (
            ae[1:-1, 1:-1] * correction[2:, 1:-1]
            + aw[1:-1, 1:-1] * correction[:-2, 1:-1]
            + an[1:-1, 1:-1] * correction[1:-1, 2:]
            + ass[1:-1, 1:-1] * correction[1:-1, :-2] + defect[1:-1, 1:-1]
        ) / ap[1:-1, 1:-1]
        correction[1, 1] = 0.0


def correct_fields(state, correction, grid, pressure_relaxation, du, dv) -> None:
    """Apply SIMPLE pressure and face-velocity corrections."""
    state.pressure[1:-1, 1:-1] += pressure_relaxation * correction[1:-1, 1:-1]
    state.u[1:-1, 1:-1] += du * (correction[1:-2, 1:-1] - correction[2:-1, 1:-1])
    state.v[1:-1, 1:-1] += dv * (correction[1:-1, 1:-2] - correction[1:-1, 2:-1])


def continuity_residual(state, grid, density, xp):
    """Return the maximum absolute cell mass imbalance."""
    imbalance = density * (
        (state.u[:-1, 1:-1] - state.u[1:, 1:-1]) * grid.dy
        + (state.v[1:-1, :-1] - state.v[1:-1, 1:]) * grid.dx
    )
    return xp.max(xp.abs(imbalance))

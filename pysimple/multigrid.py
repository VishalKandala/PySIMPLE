"""Array-backend-generic geometric multigrid for five-point equations."""


def _apply_reference(field, apply_boundaries):
    apply_boundaries(field)
    field[1, 1] = 0.0


def _smooth(field, coefficients, rhs, iterations, apply_boundaries):
    ae, aw, an, ass, ap = coefficients
    for _ in range(iterations):
        _apply_reference(field, apply_boundaries)
        field[1:-1, 1:-1] = (
            ae[1:-1, 1:-1] * field[2:, 1:-1]
            + aw[1:-1, 1:-1] * field[:-2, 1:-1]
            + an[1:-1, 1:-1] * field[1:-1, 2:]
            + ass[1:-1, 1:-1] * field[1:-1, :-2] + rhs[1:-1, 1:-1]
        ) / ap[1:-1, 1:-1]
        field[1, 1] = 0.0


def _residual(field, coefficients, rhs, xp):
    ae, aw, an, ass, ap = coefficients
    residual = xp.zeros_like(field)
    residual[1:-1, 1:-1] = rhs[1:-1, 1:-1] - (
        ap[1:-1, 1:-1] * field[1:-1, 1:-1]
        - ae[1:-1, 1:-1] * field[2:, 1:-1] - aw[1:-1, 1:-1] * field[:-2, 1:-1]
        - an[1:-1, 1:-1] * field[1:-1, 2:] - ass[1:-1, 1:-1] * field[1:-1, :-2]
    )
    residual[1, 1] = 0.0
    return residual


def _restrict(fine, xp):
    nx, ny = fine.shape[0] - 2, fine.shape[1] - 2
    if nx < 4 or ny < 4 or nx % 2 or ny % 2:
        return None
    coarse = xp.zeros((nx // 2 + 2, ny // 2 + 2), dtype=fine.dtype)
    coarse[1:-1, 1:-1] = fine[1:-1, 1:-1].reshape(nx // 2, 2, ny // 2, 2).mean(axis=(1, 3))
    return coarse


def _prolong(coarse, fine_shape, xp):
    fine = xp.zeros(fine_shape, dtype=coarse.dtype)
    fine[1:-1, 1:-1] = xp.repeat(xp.repeat(coarse[1:-1, 1:-1], 2, axis=0), 2, axis=1)
    return fine


def _coarse_coefficients(coefficients, xp):
    result = []
    for coefficient in coefficients:
        restricted = _restrict(coefficient, xp)
        if restricted is None:
            return None
        result.append(restricted)
    ae, aw, an, ass, _ = result
    return ae, aw, an, ass, ae + aw + an + ass


def _v_cycle(field, coefficients, rhs, apply_boundaries, xp):
    _smooth(field, coefficients, rhs, 3, apply_boundaries)
    residual = _residual(field, coefficients, rhs, xp)
    coarse_rhs = _restrict(residual, xp)
    coarse_coefficients = _coarse_coefficients(coefficients, xp)
    if coarse_rhs is None or coarse_coefficients is None:
        _smooth(field, coefficients, rhs, 20, apply_boundaries)
        return
    error = xp.zeros_like(coarse_rhs)
    _v_cycle(error, coarse_coefficients, coarse_rhs, apply_boundaries, xp)
    field += _prolong(error, field.shape, xp)
    _smooth(field, coefficients, rhs, 3, apply_boundaries)


def solve_pressure_multigrid(field, coefficients, rhs, cycles, apply_boundaries, xp):
    """Apply V-cycles to ``A field = rhs`` using backend vector operations only."""
    for _ in range(cycles):
        _v_cycle(field, coefficients, rhs, apply_boundaries, xp)
    _apply_reference(field, apply_boundaries)

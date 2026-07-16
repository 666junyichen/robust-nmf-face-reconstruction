"""Nonnegative matrix factorization algorithms."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class NMFResult:
    """Result of a nonnegative matrix factorization."""

    basis: NDArray[np.float64]
    coefficients: NDArray[np.float64]
    loss_history: NDArray[np.float64]
    n_iter: int
    converged: bool
    residual: NDArray[np.float64] | None = None

    @property
    def reconstruction(self) -> NDArray[np.float64]:
        """Return the low-rank component of the fitted model."""
        return self.basis @ self.coefficients


def _validate_common(
    data: NDArray[np.generic],
    rank: int,
    max_iter: int,
    tol: float,
    epsilon: float,
) -> NDArray[np.float64]:
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy ndarray")
    if data.ndim != 2:
        raise ValueError("data must be a 2D array")
    if data.size == 0:
        raise ValueError("data must be nonempty")
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError("data must be numeric")
    if np.issubdtype(data.dtype, np.complexfloating):
        raise TypeError("data must be real-valued")
    if not np.all(np.isfinite(data)):
        raise ValueError("data must contain only finite values")
    if np.any(data < 0):
        raise ValueError("data must be nonnegative")
    if isinstance(rank, (bool, np.bool_)) or not isinstance(rank, Integral):
        raise TypeError("rank must be an integer")
    if not 1 <= rank <= min(data.shape):
        raise ValueError("rank must be between 1 and min(data.shape)")
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, Integral):
        raise TypeError("max_iter must be a positive integer")
    if max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    _validate_real_option("tol", tol, minimum=0.0, strict=False)
    _validate_real_option("epsilon", epsilon, minimum=0.0, strict=True)
    if epsilon > 1e-6:
        raise ValueError("epsilon must be at most 1e-6")

    converted = np.array(data, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(converted)):
        raise ValueError("data must remain finite when converted to float64")
    return converted


def _validate_real_option(
    name: str, value: float, *, minimum: float, strict: bool
) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if (strict and value <= minimum) or (not strict and value < minimum):
        relation = "greater than" if strict else "at least"
        raise ValueError(f"{name} must be {relation} {minimum}")


def _initial_factors(
    data: NDArray[np.float64], rank: int, seed: int | None, epsilon: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    basis = np.maximum(rng.random((data.shape[0], rank)), epsilon)
    coefficients = np.maximum(rng.random((rank, data.shape[1])), epsilon)
    mean_reconstruction = float(np.mean(basis @ coefficients))
    if mean_reconstruction > epsilon:
        coefficients *= float(np.mean(data)) / mean_reconstruction
    _normalize_basis(basis, coefficients, epsilon)
    return basis, coefficients


def _normalize_basis(
    basis: NDArray[np.float64],
    coefficients: NDArray[np.float64],
    epsilon: float,
) -> None:
    scales = np.linalg.norm(basis, axis=0)
    usable_scales = np.where(scales > epsilon, scales, 1.0)
    basis /= usable_scales
    coefficients *= usable_scales[:, None]


def _has_converged(previous: float, current: float, tol: float, epsilon: float) -> bool:
    improvement = previous - current
    return improvement >= 0.0 and improvement <= tol * max(abs(previous), epsilon)


def fit_l2_nmf(
    data: NDArray[np.generic],
    rank: int,
    *,
    max_iter: int = 500,
    tol: float = 1e-5,
    seed: int | None = None,
    epsilon: float = 1e-10,
) -> NMFResult:
    """Fit Euclidean NMF using multiplicative updates.

    Internally, nonzero data is divided by its maximum. ``loss_history`` is
    therefore the scale-invariant objective
    ``0.5 * ||data / max(data) - basis @ scaled_coefficients||_F**2``.
    Returned factors split the original data scale between them to avoid
    overflowing either factor. Basis columns are normalized during fitting.
    ``epsilon`` is restricted to ``(0, 1e-6]`` and acts only as a numerical
    floor.
    """
    matrix = _validate_common(data, rank, max_iter, tol, epsilon)
    if not np.any(matrix):
        return _zero_result(matrix.shape, rank, robust=False)

    data_scale = float(np.max(matrix))
    scaled_matrix = matrix / data_scale
    basis, coefficients = _initial_factors(scaled_matrix, rank, seed, epsilon)
    loss_history = [_frobenius_loss(scaled_matrix - basis @ coefficients)]
    converged = False

    for _ in range(max_iter):
        coefficients *= _stable_ratio(
            basis.T @ scaled_matrix,
            (basis.T @ basis) @ coefficients,
            epsilon,
        )
        basis *= _stable_ratio(
            scaled_matrix @ coefficients.T,
            basis @ (coefficients @ coefficients.T),
            epsilon,
        )
        _normalize_basis(basis, coefficients, epsilon)

        current_loss = _frobenius_loss(scaled_matrix - basis @ coefficients)
        loss_history.append(current_loss)
        if _has_converged(loss_history[-2], current_loss, tol, epsilon):
            converged = True
            break

    _restore_factor_scale(basis, coefficients, data_scale)
    return NMFResult(
        basis=basis,
        coefficients=coefficients,
        loss_history=np.asarray(loss_history),
        n_iter=len(loss_history) - 1,
        converged=converged,
    )


def fit_l21_nmf(
    data: NDArray[np.generic],
    rank: int,
    *,
    regularization: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-5,
    seed: int | None = None,
    epsilon: float = 1e-10,
) -> NMFResult:
    """Fit NMF with a column-sparse explicit residual.

    Nonzero data is divided by its maximum. Alternating updates reduce the
    scale-invariant objective ``0.5 * ||scaled_data - W H - E||_F**2`` plus
    ``regularization * sum_j ||E[:, j]||_2``. Thus ``loss_history`` and the
    meaning of ``regularization`` do not depend on input units. The residual
    update is exact column-wise group soft-thresholding; the factor update is a
    multiplicative Euclidean NMF step on ``scaled_data - E``, which remains
    nonnegative. Returned factors split the original data scale between them,
    and the residual is safely restored to original units. ``epsilon`` is
    restricted to ``(0, 1e-6]`` and acts only as a numerical floor. Global
    convergence is not claimed.
    """
    matrix = _validate_common(data, rank, max_iter, tol, epsilon)
    _validate_real_option(
        "regularization", regularization, minimum=0.0, strict=True
    )
    if not np.any(matrix):
        return _zero_result(matrix.shape, rank, robust=True)

    data_scale = float(np.max(matrix))
    scaled_matrix = matrix / data_scale
    basis, coefficients = _initial_factors(scaled_matrix, rank, seed, epsilon)
    residual = _threshold_residual(
        scaled_matrix - basis @ coefficients, regularization
    )
    loss_history = [
        _robust_loss(scaled_matrix, basis, coefficients, residual, regularization)
    ]
    converged = False

    for _ in range(max_iter):
        adjusted = scaled_matrix - residual
        coefficients *= _stable_ratio(
            basis.T @ adjusted,
            (basis.T @ basis) @ coefficients,
            epsilon,
        )
        basis *= _stable_ratio(
            adjusted @ coefficients.T,
            basis @ (coefficients @ coefficients.T),
            epsilon,
        )
        _normalize_basis(basis, coefficients, epsilon)

        residual = _threshold_residual(
            scaled_matrix - basis @ coefficients, regularization
        )
        current_loss = _robust_loss(
            scaled_matrix, basis, coefficients, residual, regularization
        )
        loss_history.append(current_loss)
        if _has_converged(loss_history[-2], current_loss, tol, epsilon):
            converged = True
            break

    _restore_factor_scale(basis, coefficients, data_scale)
    _restore_residual_scale(residual, data_scale)
    return NMFResult(
        basis=basis,
        coefficients=coefficients,
        loss_history=np.asarray(loss_history),
        n_iter=len(loss_history) - 1,
        converged=converged,
        residual=residual,
    )


def _frobenius_loss(error: NDArray[np.float64]) -> float:
    return 0.5 * float(np.sum(error * error))


def _stable_ratio(
    numerator: NDArray[np.float64],
    denominator: NDArray[np.float64],
    epsilon: float,
) -> NDArray[np.float64]:
    """Return an elementwise ratio, flooring only tiny denominators."""
    return numerator / np.maximum(denominator, epsilon)


def _restore_factor_scale(
    basis: NDArray[np.float64],
    coefficients: NDArray[np.float64],
    data_scale: float,
) -> None:
    """Split the data scale across factors to keep both representable."""
    normalized_max = float(np.max(basis @ coefficients))
    accumulation_guard = 1.0 - (basis.shape[1] + 2) * np.finfo(np.float64).eps
    safe_scale = np.finfo(np.float64).max
    if normalized_max > accumulation_guard:
        safe_scale *= accumulation_guard / normalized_max
    restored_scale = min(data_scale, safe_scale)
    basis_scale = float(np.sqrt(restored_scale))
    basis *= basis_scale
    coefficients *= restored_scale / basis_scale


def _restore_residual_scale(
    residual: NDArray[np.float64], data_scale: float
) -> None:
    """Restore residual units without overflowing finite float64 values."""
    if data_scale > 1.0:
        rescalable_limit = np.finfo(np.float64).max / data_scale
        np.clip(residual, -rescalable_limit, rescalable_limit, out=residual)
    residual *= data_scale


def _threshold_residual(
    error: NDArray[np.float64], regularization: float
) -> NDArray[np.float64]:
    norms = np.linalg.norm(error, axis=0)
    shrinkage = np.zeros_like(norms)
    active = norms > regularization
    shrinkage[active] = 1.0 - regularization / norms[active]
    return error * shrinkage[None, :]


def _robust_loss(
    data: NDArray[np.float64],
    basis: NDArray[np.float64],
    coefficients: NDArray[np.float64],
    residual: NDArray[np.float64],
    regularization: float,
) -> float:
    unexplained = data - basis @ coefficients - residual
    return _frobenius_loss(unexplained) + regularization * float(
        np.sum(np.linalg.norm(residual, axis=0))
    )


def _zero_result(
    shape: tuple[int, int], rank: int, *, robust: bool
) -> NMFResult:
    basis = np.zeros((shape[0], rank), dtype=np.float64)
    coefficients = np.zeros((rank, shape[1]), dtype=np.float64)
    residual = np.zeros(shape, dtype=np.float64) if robust else None
    return NMFResult(
        basis=basis,
        coefficients=coefficients,
        loss_history=np.zeros(1, dtype=np.float64),
        n_iter=0,
        converged=True,
        residual=residual,
    )

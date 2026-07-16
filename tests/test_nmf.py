from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from robust_nmf.nmf import (
    NMFResult,
    _threshold_residual,
    fit_l2_nmf,
    fit_l21_nmf,
)


def test_public_api_is_importable() -> None:
    assert NMFResult is not None
    assert callable(fit_l2_nmf)
    assert callable(fit_l21_nmf)


def test_public_api_is_exported_from_package() -> None:
    from robust_nmf import NMFResult as ExportedResult
    from robust_nmf import fit_l2_nmf as exported_l2
    from robust_nmf import fit_l21_nmf as exported_l21

    assert ExportedResult is NMFResult
    assert exported_l2 is fit_l2_nmf
    assert exported_l21 is fit_l21_nmf


def test_result_is_frozen_and_reconstruction_excludes_residual() -> None:
    basis = np.array([[1.0], [2.0]])
    coefficients = np.array([[3.0, 4.0]])
    residual = np.full((2, 2), 100.0)
    result = NMFResult(basis, coefficients, np.array([1.0]), 0, False, residual)

    np.testing.assert_array_equal(result.reconstruction, basis @ coefficients)
    with pytest.raises(FrozenInstanceError):
        result.n_iter = 2  # type: ignore[misc]


def test_l2_factorization_has_valid_shapes_and_improves_low_rank_fit() -> None:
    left = np.array([[1.0, 0.2], [0.3, 1.4], [1.2, 0.7], [0.5, 1.1]])
    right = np.array([[1.0, 0.4, 1.3, 0.2, 0.8], [0.2, 1.1, 0.5, 1.4, 0.6]])
    data = left @ right

    result = fit_l2_nmf(data, 2, max_iter=400, tol=1e-8, seed=12)

    assert result.basis.shape == (4, 2)
    assert result.coefficients.shape == (2, 5)
    assert result.reconstruction.shape == data.shape
    assert result.residual is None
    assert result.loss_history.shape == (result.n_iter + 1,)
    assert result.loss_history[-1] < result.loss_history[0] * 0.02
    assert np.count_nonzero(np.diff(result.loss_history) > 1e-10) == 0
    assert np.all(np.isfinite(result.loss_history))
    assert np.all(np.isfinite(result.basis))
    assert np.all(np.isfinite(result.coefficients))
    assert np.all(result.basis >= 0.0)
    assert np.all(result.coefficients >= 0.0)


@pytest.mark.parametrize("fit", [fit_l2_nmf, fit_l21_nmf])
def test_same_seed_is_reproducible_and_input_is_not_mutated(fit: object) -> None:
    data = np.arange(1.0, 21.0).reshape(4, 5)
    original = data.copy()

    first = fit(data, 2, max_iter=20, seed=3)  # type: ignore[operator]
    second = fit(data, 2, max_iter=20, seed=3)  # type: ignore[operator]

    np.testing.assert_array_equal(first.basis, second.basis)
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    np.testing.assert_array_equal(first.loss_history, second.loss_history)
    np.testing.assert_array_equal(data, original)


def test_l21_returns_finite_signed_residual_on_sparse_outliers() -> None:
    clean = np.outer(np.array([1.0, 0.6, 1.4, 0.8]), np.linspace(0.5, 1.5, 7))
    contaminated = clean.copy()
    contaminated[:, 2] += np.array([15.0, 0.0, 8.0, 3.0])
    contaminated[:, 5] += np.array([0.0, 12.0, 0.0, 6.0])

    result = fit_l21_nmf(
        contaminated,
        1,
        regularization=0.8,
        max_iter=200,
        tol=1e-8,
        seed=7,
    )

    assert result.residual is not None
    assert result.residual.shape == contaminated.shape
    assert result.loss_history.shape == (result.n_iter + 1,)
    assert np.all(np.isfinite(result.residual))
    assert np.all(np.isfinite(result.basis))
    assert np.all(np.isfinite(result.coefficients))
    assert np.all(result.basis >= 0.0)
    assert np.all(result.coefficients >= 0.0)
    assert result.loss_history[-1] < result.loss_history[0]


@pytest.mark.parametrize("fit", [fit_l2_nmf, fit_l21_nmf])
def test_zero_matrix_is_handled_without_nan_or_division_errors(fit: object) -> None:
    result = fit(np.zeros((3, 4)), 2, max_iter=10, seed=1)  # type: ignore[operator]

    np.testing.assert_array_equal(result.reconstruction, np.zeros((3, 4)))
    np.testing.assert_array_equal(result.loss_history, np.zeros(result.n_iter + 1))
    assert np.all(np.isfinite(result.basis))
    assert np.all(np.isfinite(result.coefficients))


@pytest.mark.parametrize("fit", [fit_l2_nmf, fit_l21_nmf])
def test_large_finite_data_produces_finite_result(fit: object) -> None:
    data = np.full((2, 2), 1e200)

    result = fit(data, 2, max_iter=20, seed=5)  # type: ignore[operator]

    assert np.all(np.isfinite(result.basis))
    assert np.all(np.isfinite(result.coefficients))
    assert np.all(np.isfinite(result.reconstruction))
    assert np.all(np.isfinite(result.loss_history))
    if result.residual is not None:
        assert np.all(np.isfinite(result.residual))


@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("fit", [fit_l2_nmf, fit_l21_nmf])
def test_max_float_data_produces_finite_result_without_warnings(
    fit: object, rank: int
) -> None:
    data = np.full((2, 2), np.finfo(np.float64).max)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = fit(data, rank, max_iter=20, seed=5)  # type: ignore[operator]
        reconstruction = result.reconstruction

    assert np.all(np.isfinite(result.basis))
    assert np.all(np.isfinite(result.coefficients))
    assert np.all(np.isfinite(reconstruction))
    assert np.all(np.isfinite(result.loss_history))
    if result.residual is not None:
        assert np.all(np.isfinite(result.residual))


@pytest.mark.parametrize("fit", [fit_l2_nmf, fit_l21_nmf])
def test_normalized_loss_history_is_scale_invariant(fit: object) -> None:
    data = np.array([[1.0, 0.3], [0.4, 0.8]])

    ordinary = fit(data, 1, max_iter=12, tol=0.0, seed=8)  # type: ignore[operator]
    scaled = fit(data * 1e150, 1, max_iter=12, tol=0.0, seed=8)  # type: ignore[operator]

    np.testing.assert_allclose(ordinary.loss_history, scaled.loss_history)


def test_group_threshold_uses_exact_column_norms() -> None:
    error = np.array([[3e-8, 0.0, 0.0], [4e-8, 0.0, 2e-8]])
    expected = np.array([[2.4e-8, 0.0, 0.0], [3.2e-8, 0.0, 1e-8]])

    result = _threshold_residual(error, regularization=1e-8)

    np.testing.assert_allclose(result, expected, rtol=1e-14, atol=0.0)


@pytest.mark.parametrize(
    ("data", "error", "match"),
    [
        ([[1.0]], TypeError, "ndarray"),
        (np.array([1.0]), ValueError, "2D"),
        (np.empty((0, 2)), ValueError, "nonempty"),
        (np.array([["x"]]), TypeError, "numeric"),
        (np.array([[1.0 + 2.0j]]), TypeError, "real"),
        (np.array([[np.nan]]), ValueError, "finite"),
        (np.array([[np.inf]]), ValueError, "finite"),
        (np.array([[-0.1]]), ValueError, "nonnegative"),
    ],
)
@pytest.mark.parametrize("fit", [fit_l2_nmf, fit_l21_nmf])
def test_rejects_invalid_data(
    fit: object, data: object, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        fit(data, 1)  # type: ignore[operator]


@pytest.mark.parametrize("rank", [True, 0, -1, 1.5, 4])
@pytest.mark.parametrize("fit", [fit_l2_nmf, fit_l21_nmf])
def test_rejects_invalid_rank(fit: object, rank: object) -> None:
    with pytest.raises((TypeError, ValueError), match="rank"):
        fit(np.ones((2, 3)), rank)  # type: ignore[operator]


@pytest.mark.parametrize("max_iter", [True, 0, -1, 2.5])
@pytest.mark.parametrize("fit", [fit_l2_nmf, fit_l21_nmf])
def test_rejects_invalid_max_iter(fit: object, max_iter: object) -> None:
    with pytest.raises((TypeError, ValueError), match="max_iter"):
        fit(np.ones((2, 3)), 1, max_iter=max_iter)  # type: ignore[operator]


@pytest.mark.parametrize(("name", "value"), [("tol", -1.0), ("tol", np.nan), ("epsilon", 0.0), ("epsilon", np.inf)])
@pytest.mark.parametrize("fit", [fit_l2_nmf, fit_l21_nmf])
def test_rejects_invalid_common_numeric_options(
    fit: object, name: str, value: float
) -> None:
    with pytest.raises((TypeError, ValueError), match=name):
        fit(np.ones((2, 3)), 1, **{name: value})  # type: ignore[operator]


@pytest.mark.parametrize("fit", [fit_l2_nmf, fit_l21_nmf])
def test_epsilon_accepts_documented_upper_boundary(fit: object) -> None:
    result = fit(  # type: ignore[operator]
        np.ones((2, 2)), 1, epsilon=1e-6, max_iter=1
    )

    assert np.all(np.isfinite(result.loss_history))


@pytest.mark.parametrize("epsilon", [np.nextafter(1e-6, np.inf), 1.0])
@pytest.mark.parametrize("fit", [fit_l2_nmf, fit_l21_nmf])
def test_rejects_epsilon_above_stability_contract(
    fit: object, epsilon: float
) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        fit(np.ones((2, 2)), 1, epsilon=epsilon)  # type: ignore[operator]


@pytest.mark.parametrize("regularization", [0.0, -1.0, np.nan, np.inf, "1"])
def test_l21_rejects_invalid_regularization(regularization: object) -> None:
    with pytest.raises((TypeError, ValueError), match="regularization"):
        fit_l21_nmf(np.ones((2, 3)), 1, regularization=regularization)  # type: ignore[arg-type]

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from robust_nmf.metrics import (
    ClusteringMetrics,
    evaluate_clustering,
    relative_reconstruction_error,
)


@pytest.fixture(autouse=True)
def _limit_joblib_cpu_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "1")


def test_relative_reconstruction_error_matches_frobenius_definition() -> None:
    clean = np.array([[3.0, 4.0], [0.0, 0.0]])
    reconstructed = np.array([[0.0, 4.0], [0.0, 0.0]])

    assert relative_reconstruction_error(clean, reconstructed) == pytest.approx(0.6)


def test_relative_reconstruction_error_is_scale_safe_for_extreme_values() -> None:
    clean = np.full((2, 2), 1e300)
    reconstructed = clean * 0.5

    result = relative_reconstruction_error(clean, reconstructed)

    assert np.isfinite(result)
    assert result == pytest.approx(0.5)


def test_relative_reconstruction_error_defines_zero_reference_cases() -> None:
    assert relative_reconstruction_error(np.zeros((2, 2)), np.zeros((2, 2))) == 0.0
    with pytest.raises(ValueError, match="zero clean"):
        relative_reconstruction_error(np.zeros((2, 2)), np.ones((2, 2)))


@pytest.mark.parametrize(
    ("clean", "reconstructed", "error", "match"),
    [
        (np.ones(2), np.ones(2), ValueError, "2D"),
        (np.ones((2, 2)), np.ones((2, 3)), ValueError, "same shape"),
        (np.array([[1.0 + 1.0j]]), np.ones((1, 1)), TypeError, "real"),
        (np.array([[np.nan]]), np.ones((1, 1)), ValueError, "finite"),
        (np.array([[np.inf]]), np.ones((1, 1)), ValueError, "finite"),
    ],
)
def test_relative_reconstruction_error_rejects_invalid_arrays(
    clean: np.ndarray,
    reconstructed: np.ndarray,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        relative_reconstruction_error(clean, reconstructed)


@pytest.mark.parametrize("epsilon", [0.0, -1.0, np.nan, np.inf, "x"])
def test_relative_reconstruction_error_rejects_invalid_epsilon(epsilon: object) -> None:
    with pytest.raises((TypeError, ValueError), match="epsilon"):
        relative_reconstruction_error(np.ones((1, 1)), np.ones((1, 1)), epsilon=epsilon)  # type: ignore[arg-type]


def test_clustering_is_perfect_on_separable_noncontiguous_labels() -> None:
    coefficients = np.array(
        [
            [0.0, 0.1, 10.0, 10.1, 20.0, 20.1],
            [0.1, 0.0, 10.1, 10.0, -5.0, -5.1],
        ]
    )
    labels = np.array([-7, -7, 42, 42, 100, 100])

    result = evaluate_clustering(coefficients, labels, seed=9)

    assert result.accuracy == pytest.approx(1.0)
    assert result.nmi == pytest.approx(1.0)
    np.testing.assert_array_equal(result.predicted_labels, labels)


def test_clustering_is_deterministic_and_result_is_frozen() -> None:
    coefficients = np.array([[0.0, 0.2, 5.0, 5.2], [1.0, 1.1, 3.0, 3.1]])
    labels = np.array([10, 10, -3, -3])

    first = evaluate_clustering(coefficients, labels, seed=2)
    second = evaluate_clustering(coefficients, labels, seed=2)

    assert isinstance(first, ClusteringMetrics)
    np.testing.assert_array_equal(first.predicted_labels, second.predicted_labels)
    assert first.accuracy == second.accuracy
    assert first.nmi == second.nmi
    with pytest.raises(FrozenInstanceError):
        first.accuracy = 0.0  # type: ignore[misc]


def test_clustering_metrics_copies_predicted_labels_and_makes_them_read_only() -> None:
    predicted = np.array([4, -2], dtype=np.int64)

    metrics = ClusteringMetrics(1.0, 1.0, predicted)

    assert not metrics.predicted_labels.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        metrics.predicted_labels[0] = 9
    predicted[0] = 9
    np.testing.assert_array_equal(metrics.predicted_labels, [4, -2])


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_clustering_accepts_seed_boundaries(seed: int) -> None:
    result = evaluate_clustering(
        np.array([[0.0, 0.1, 4.0, 4.1]]),
        np.array([0, 0, 1, 1]),
        seed=seed,
    )

    assert result.accuracy == pytest.approx(1.0)


@pytest.mark.parametrize("seed", [True, -1, 2**32, 1.5])
def test_clustering_rejects_invalid_seed(seed: object) -> None:
    with pytest.raises((TypeError, ValueError), match="seed"):
        evaluate_clustering(
            np.array([[0.0, 0.1, 4.0, 4.1]]),
            np.array([0, 0, 1, 1]),
            seed=seed,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("coefficients", "labels", "error", "match"),
    [
        (np.ones(4), np.array([0, 0, 1, 1]), ValueError, "2D"),
        (np.ones((2, 4)), np.array([0, 1, 1]), ValueError, "sample"),
        (np.empty((0, 4)), np.array([0, 0, 1, 1]), ValueError, "nonempty"),
        (np.array([[0.0, np.nan]]), np.array([0, 1]), ValueError, "finite"),
        (np.array([[0.0 + 1.0j, 1.0]]), np.array([0, 1]), TypeError, "real"),
        (np.ones((2, 4)), np.array([[0, 0, 1, 1]]), ValueError, "1D"),
        (np.ones((2, 4)), np.array([0.0, 0.0, 1.0, 1.0]), TypeError, "integer"),
    ],
)
def test_clustering_rejects_invalid_inputs(
    coefficients: np.ndarray,
    labels: np.ndarray,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        evaluate_clustering(coefficients, labels)


@pytest.mark.parametrize("n_clusters", [True, 0, -1, 1, 3, 2.5])
def test_clustering_rejects_invalid_cluster_count(n_clusters: object) -> None:
    with pytest.raises((TypeError, ValueError), match="n_clusters"):
        evaluate_clustering(
            np.array([[0.0, 0.1, 4.0, 4.1]]),
            np.array([0, 0, 1, 1]),
            n_clusters=n_clusters,  # type: ignore[arg-type]
        )


def test_clustering_requires_at_least_two_classes() -> None:
    with pytest.raises(ValueError, match="at least two"):
        evaluate_clustering(np.array([[0.0, 1.0]]), np.array([5, 5]))


def test_metrics_api_is_exported_from_package() -> None:
    from robust_nmf import ClusteringMetrics as ExportedMetrics
    from robust_nmf import evaluate_clustering as exported_clustering
    from robust_nmf import relative_reconstruction_error as exported_error

    assert ExportedMetrics is ClusteringMetrics
    assert exported_clustering is evaluate_clustering
    assert exported_error is relative_reconstruction_error

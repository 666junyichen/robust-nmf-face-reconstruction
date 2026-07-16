"""Scale-safe reconstruction and clustering evaluation metrics."""

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score


@dataclass(frozen=True)
class ClusteringMetrics:
    """Accuracy, NMI, and cluster labels aligned to the true label values."""

    accuracy: float
    nmi: float
    predicted_labels: NDArray[np.integer]


def _real_finite_2d(value: object, name: str) -> NDArray[np.floating]:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D")
    if array.size == 0:
        raise ValueError(f"{name} must be nonempty")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} must contain real values")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _norm_parts(array: NDArray[np.float64]) -> tuple[float, float]:
    scale = float(np.max(np.abs(array)))
    if scale == 0.0:
        return 0.0, 0.0
    return scale, float(np.linalg.norm(array / scale))


def relative_reconstruction_error(
    clean: object,
    reconstructed: object,
    epsilon: float = np.finfo(np.float64).tiny,
) -> float:
    """Return ``||clean - reconstructed||_F / ||clean||_F`` safely.

    A clean norm no greater than ``epsilon`` is treated as zero. Two zero
    matrices have error zero; a nonzero reconstruction against a zero clean
    matrix is undefined and raises ``ValueError``. Ratios beyond float64 range
    are capped at the largest finite float.
    """

    if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
        raise TypeError("epsilon must be a positive finite real number")
    epsilon_value = float(epsilon)
    if not math.isfinite(epsilon_value) or epsilon_value <= 0.0:
        raise ValueError("epsilon must be a positive finite real number")

    clean_array = _real_finite_2d(clean, "clean")
    reconstructed_array = _real_finite_2d(reconstructed, "reconstructed")
    if clean_array.shape != reconstructed_array.shape:
        raise ValueError("clean and reconstructed must have the same shape")

    clean_scale, clean_unit_norm = _norm_parts(clean_array)
    reconstructed_scale, _ = _norm_parts(reconstructed_array)
    if clean_scale == 0.0:
        if reconstructed_scale == 0.0:
            return 0.0
        raise ValueError("relative error is undefined for zero clean data and nonzero reconstruction")

    log_clean_norm = math.log(clean_scale) + math.log(clean_unit_norm)
    if log_clean_norm <= math.log(epsilon_value):
        raise ValueError("clean norm is zero within epsilon")

    error_scale = max(clean_scale, reconstructed_scale)
    scaled_error = clean_array / error_scale - reconstructed_array / error_scale
    error_unit_norm = float(np.linalg.norm(scaled_error))
    if error_unit_norm == 0.0:
        return 0.0

    scale_ratio = error_scale / clean_scale
    if math.isfinite(scale_ratio):
        ratio = scale_ratio * error_unit_norm / clean_unit_norm
        if math.isfinite(ratio):
            return float(ratio)

    log_ratio = (
        math.log(error_scale)
        + math.log(error_unit_norm)
        - math.log(clean_scale)
        - math.log(clean_unit_norm)
    )
    maximum = np.finfo(np.float64).max
    if log_ratio >= math.log(maximum):
        return float(maximum)
    return float(math.exp(log_ratio))


def evaluate_clustering(
    coefficients: object,
    labels: object,
    *,
    n_clusters: int | None = None,
    seed: int = 0,
) -> ClusteringMetrics:
    """Cluster coefficient columns and align cluster IDs with true labels."""

    features = _real_finite_2d(coefficients, "coefficients")
    label_array = np.asarray(labels)
    if label_array.ndim != 1:
        raise ValueError("labels must be 1D")
    if not np.issubdtype(label_array.dtype, np.integer):
        raise TypeError("labels must have an integer dtype")
    if features.shape[1] != label_array.size:
        raise ValueError("coefficient sample count must match labels")
    if label_array.size == 0:
        raise ValueError("labels must be nonempty")

    unique_labels, encoded_labels = np.unique(label_array, return_inverse=True)
    class_count = len(unique_labels)
    if class_count < 2:
        raise ValueError("clustering evaluation requires at least two classes")
    if n_clusters is None:
        cluster_count = class_count
    else:
        if isinstance(n_clusters, bool) or not isinstance(n_clusters, (int, np.integer)):
            raise TypeError("n_clusters must be an integer")
        cluster_count = int(n_clusters)
        if cluster_count != class_count:
            raise ValueError("n_clusters must equal the number of unique labels")
    if cluster_count > label_array.size:
        raise ValueError("n_clusters cannot exceed the sample count")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")

    cluster_ids = KMeans(
        n_clusters=cluster_count,
        random_state=int(seed),
        n_init=10,
    ).fit_predict(features.T)

    contingency = np.zeros((cluster_count, class_count), dtype=np.int64)
    np.add.at(contingency, (cluster_ids, encoded_labels), 1)
    cluster_indices, class_indices = linear_sum_assignment(-contingency)
    mapping = np.empty(cluster_count, dtype=unique_labels.dtype)
    mapping[cluster_indices] = unique_labels[class_indices]
    predicted_labels = mapping[cluster_ids]
    predicted_labels.setflags(write=False)

    accuracy = float(np.mean(predicted_labels == label_array))
    nmi = float(normalized_mutual_info_score(label_array, cluster_ids))
    return ClusteringMetrics(accuracy, nmi, predicted_labels)


__all__ = [
    "ClusteringMetrics",
    "evaluate_clustering",
    "relative_reconstruction_error",
]

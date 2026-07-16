"""Robust NMF tools for face reconstruction experiments."""

from .data import FaceDataset, load_face_directory
from .metrics import (
    ClusteringMetrics,
    evaluate_clustering,
    relative_reconstruction_error,
)
from .nmf import NMFResult, fit_l2_nmf, fit_l21_nmf

__version__ = "0.1.0"

__all__ = [
    "ClusteringMetrics",
    "FaceDataset",
    "NMFResult",
    "evaluate_clustering",
    "fit_l2_nmf",
    "fit_l21_nmf",
    "load_face_directory",
    "relative_reconstruction_error",
]

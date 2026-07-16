"""Noise models for image reconstruction experiments."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np


def add_salt_pepper_noise(
    image: np.ndarray,
    corruption: float,
    salt_ratio: float,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Return a copy of ``image`` with reproducible salt-and-pepper noise.

    Exactly ``round(image.size * corruption)`` unique positions are selected.
    Of those positions, ``round(selected_count * salt_ratio)`` are set to 1.0
    and the rest to 0.0. If a selected input value is already equal to its
    assigned value, fewer array values will differ even though the requested
    number of positions was selected.

    Args:
        image: A nonempty, finite, real-valued numeric NumPy array.
        corruption: Fraction of array positions to select, in ``[0, 1]``.
        salt_ratio: Fraction of selected positions to set to 1.0, in
            ``[0, 1]``. The remaining selected positions are set to 0.0.
        seed: Optional seed passed to :func:`numpy.random.default_rng`.

    Returns:
        A working copy normalized to NumPy ``float64`` dtype with the same
        shape as ``image``.

    Raises:
        TypeError: If ``image`` is not a real-valued numeric NumPy array or a
            rate is not a real number.
        ValueError: If ``image`` is empty or non-finite, or a rate is outside
            the closed unit interval.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.size == 0:
        raise ValueError("image must be nonempty")
    if not np.issubdtype(image.dtype, np.number) or np.issubdtype(
        image.dtype, np.complexfloating
    ):
        raise TypeError("image must contain real-valued numeric data")
    if not np.all(np.isfinite(image)):
        raise ValueError("image values must all be finite")

    normalized_rates: list[float] = []
    for name, rate in (("corruption", corruption), ("salt_ratio", salt_ratio)):
        if isinstance(rate, (bool, np.bool_)) or not isinstance(rate, Real):
            raise TypeError(f"{name} must be a real number")
        try:
            normalized_rate = float(rate)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"{name} must be finite and in [0, 1]") from exc
        if not math.isfinite(normalized_rate) or not 0.0 <= normalized_rate <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
        normalized_rates.append(normalized_rate)

    corruption, salt_ratio = normalized_rates

    result = image.astype(float, copy=True)
    changed_count = round(image.size * corruption)
    if changed_count == 0:
        return result

    rng = np.random.default_rng(seed)
    selected = rng.choice(image.size, size=changed_count, replace=False)
    salt_count = round(changed_count * salt_ratio)
    result.flat[selected[:salt_count]] = 1.0
    result.flat[selected[salt_count:]] = 0.0
    return result

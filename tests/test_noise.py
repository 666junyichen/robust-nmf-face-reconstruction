import numpy as np
import pytest

from robust_nmf.noise import add_salt_pepper_noise


def test_returns_float_copy_with_same_shape() -> None:
    image = np.arange(24).reshape(2, 3, 4)
    original = image.copy()

    result = add_salt_pepper_noise(image, 0.0, 0.5)

    assert result.shape == image.shape
    assert np.issubdtype(result.dtype, np.floating)
    assert result is not image
    np.testing.assert_array_equal(result, original)
    np.testing.assert_array_equal(image, original)


def test_corrupts_exact_rounded_count_with_exact_salt_split() -> None:
    image = np.full((4, 5), 0.5)
    original = image.copy()

    result = add_salt_pepper_noise(image, 0.26, 0.4, seed=17)

    assert np.count_nonzero(result != image) == 5
    assert np.count_nonzero(result == 1.0) == 2
    assert np.count_nonzero(result == 0.0) == 3
    np.testing.assert_array_equal(image, original)


def test_same_seed_is_reproducible() -> None:
    image = np.full((10, 10), 0.5)

    first = add_salt_pepper_noise(image, 0.4, 0.25, seed=42)
    second = add_salt_pepper_noise(image, 0.4, 0.25, seed=42)

    np.testing.assert_array_equal(first, second)


def test_different_seeds_select_different_pixels() -> None:
    image = np.full((10, 10), 0.5)

    first = add_salt_pepper_noise(image, 0.4, 0.5, seed=1)
    second = add_salt_pepper_noise(image, 0.4, 0.5, seed=2)

    assert not np.array_equal(first, second)


@pytest.mark.parametrize(
    ("corruption", "salt_ratio", "expected_zeros", "expected_ones"),
    [
        (1.0, 0.0, 12, 0),
        (1.0, 1.0, 0, 12),
    ],
)
def test_full_corruption_honors_extreme_salt_ratios(
    corruption: float,
    salt_ratio: float,
    expected_zeros: int,
    expected_ones: int,
) -> None:
    image = np.full((3, 4), 0.5)

    result = add_salt_pepper_noise(image, corruption, salt_ratio, seed=9)

    assert np.count_nonzero(result == 0.0) == expected_zeros
    assert np.count_nonzero(result == 1.0) == expected_ones


def test_accepts_zero_dimensional_numeric_array() -> None:
    image = np.array(0.5)

    result = add_salt_pepper_noise(image, 1.0, 1.0, seed=3)

    assert result.shape == ()
    assert result.item() == 1.0


@pytest.mark.parametrize(
    ("image", "error", "match"),
    [
        ([0.5], TypeError, "ndarray"),
        (np.array([]), ValueError, "nonempty"),
        (np.array(["pixel"]), TypeError, "numeric"),
        (np.array([0.0, np.nan]), ValueError, "finite"),
        (np.array([0.0, np.inf]), ValueError, "finite"),
    ],
)
def test_rejects_invalid_images(
    image: object,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        add_salt_pepper_noise(image, 0.5, 0.5)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("corruption", "salt_ratio", "match"),
    [
        (-0.01, 0.5, "corruption"),
        (1.01, 0.5, "corruption"),
        (np.nan, 0.5, "corruption"),
        (0.5, -0.01, "salt_ratio"),
        (0.5, 1.01, "salt_ratio"),
        (0.5, np.inf, "salt_ratio"),
    ],
)
def test_rejects_rates_outside_closed_unit_interval(
    corruption: float,
    salt_ratio: float,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        add_salt_pepper_noise(np.ones(4), corruption, salt_ratio)


@pytest.mark.parametrize(
    ("corruption", "salt_ratio", "match"),
    [
        ("0.5", 0.5, "corruption"),
        (0.5, "0.5", "salt_ratio"),
    ],
)
def test_rejects_nonnumeric_rates(
    corruption: object,
    salt_ratio: object,
    match: str,
) -> None:
    with pytest.raises(TypeError, match=match):
        add_salt_pepper_noise(
            np.ones(4),
            corruption,  # type: ignore[arg-type]
            salt_ratio,  # type: ignore[arg-type]
        )

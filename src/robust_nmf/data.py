"""Deterministic loading of directory-organized grayscale face images."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError


_SUPPORTED_EXTENSIONS = {".pgm", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class FaceDataset:
    """A face dataset whose flattened images occupy matrix columns."""

    matrix: NDArray[np.float64]
    labels: NDArray[np.integer]
    image_shape: tuple[int, int]
    paths: tuple[Path, ...]
    class_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.matrix, np.ndarray) or self.matrix.ndim != 2:
            raise TypeError("matrix must be a 2D ndarray")
        if self.matrix.dtype != np.float64:
            raise TypeError("matrix must have float64 dtype")
        if not np.all(np.isfinite(self.matrix)):
            raise ValueError("matrix values must be finite")
        if np.any((self.matrix < 0.0) | (self.matrix > 1.0)):
            raise ValueError("matrix values must lie in [0, 1]")
        if not isinstance(self.labels, np.ndarray) or self.labels.ndim != 1:
            raise TypeError("labels must be a 1D ndarray")
        if not np.issubdtype(self.labels.dtype, np.integer):
            raise TypeError("labels must have an integer dtype")
        if (
            not isinstance(self.image_shape, tuple)
            or len(self.image_shape) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in self.image_shape)
            or any(value <= 0 for value in self.image_shape)
        ):
            raise ValueError("image_shape must contain two positive integers")
        if self.matrix.shape[0] != self.image_shape[0] * self.image_shape[1]:
            raise ValueError("matrix pixels must match image_shape")
        sample_count = self.matrix.shape[1]
        if len(self.labels) != sample_count or len(self.paths) != sample_count:
            raise ValueError("matrix samples, labels, and paths must have equal lengths")
        if any(not isinstance(path, Path) for path in self.paths):
            raise TypeError("paths must contain Path objects")
        if not self.class_names or any(not isinstance(name, str) or not name for name in self.class_names):
            raise ValueError("class_names must contain nonempty strings")
        if sample_count and (np.min(self.labels) < 0 or np.max(self.labels) >= len(self.class_names)):
            raise ValueError("labels must index class_names")
        matrix = self.matrix.copy()
        labels = self.labels.copy()
        matrix.setflags(write=False)
        labels.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "labels", labels)


def _validate_resize(resize: tuple[int, int] | None) -> None:
    if resize is None:
        return
    if (
        not isinstance(resize, tuple)
        or len(resize) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in resize)
    ):
        raise TypeError("resize must be a (height, width) tuple of integers")
    if any(value <= 0 for value in resize):
        raise ValueError("resize dimensions must be positive")


def _image_paths(root: Path) -> tuple[Path, ...]:
    candidates = tuple(root.rglob("*"))
    if not candidates:
        raise ValueError(f"face directory is empty: {root}")
    paths = [
        path
        for path in candidates
        if path.is_file()
        and not path.name.startswith("._")
        and path.suffix.lower() in _SUPPORTED_EXTENSIONS
    ]
    if not paths:
        raise ValueError(f"no supported image files found in {root}")
    sorted_paths = tuple(
        sorted(paths, key=lambda path: (path.as_posix().casefold(), path.as_posix()))
    )
    invalid_paths = [
        path.relative_to(root).as_posix()
        for path in sorted_paths
        if len(path.relative_to(root).parts) != 2
    ]
    if invalid_paths:
        formatted = ", ".join(invalid_paths)
        raise ValueError(
            "supported images must use the root/class/image layout; "
            f"offending paths: {formatted}"
        )
    return sorted_paths


def load_face_directory(
    root: str | Path, resize: tuple[int, int] | None = None
) -> FaceDataset:
    """Load recursively discovered face images without modifying the source files.

    The supported layout is exactly ``root/class/image``; deeper nesting is
    rejected because it makes immediate-parent class labels ambiguous.
    ``resize`` is expressed as ``(height, width)``. Images are normalized to
    ``[0, 1]`` and flattened in row-major order into matrix columns.
    """

    _validate_resize(resize)
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"face directory does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"face root is not a directory: {root_path}")

    paths = _image_paths(root_path)
    class_names = tuple(sorted({path.parent.name for path in paths}))
    class_to_label = {name: index for index, name in enumerate(class_names)}
    labels = np.fromiter(
        (class_to_label[path.parent.name] for path in paths),
        dtype=np.int64,
        count=len(paths),
    )
    columns: list[NDArray[np.float64]] = []
    image_shape: tuple[int, int] | None = resize

    for path in paths:
        try:
            with Image.open(path) as source:
                image = source.convert("L")
                if resize is not None:
                    image = image.resize((resize[1], resize[0]), Image.Resampling.BILINEAR)
                current_shape = (image.height, image.width)
                if image_shape is None:
                    image_shape = current_shape
                elif current_shape != image_shape:
                    raise ValueError(
                        "inconsistent image sizes; provide resize=(height, width) "
                        f"to normalize them ({path}: {current_shape}, expected {image_shape})"
                    )
                array = np.asarray(image, dtype=np.float64) / 255.0
                columns.append(array.reshape(-1))
        except ValueError:
            raise
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"corrupt or unreadable image: {path}") from exc

    assert image_shape is not None
    matrix = np.column_stack(columns).astype(np.float64, copy=False)
    return FaceDataset(matrix, labels, image_shape, paths, class_names)


__all__ = ["FaceDataset", "load_face_directory"]

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from robust_nmf.data import FaceDataset, load_face_directory


def _save_gray(path: Path, values: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(values, dtype=np.uint8)).save(path)


def test_loader_sorts_paths_and_assigns_sorted_parent_labels(tmp_path: Path) -> None:
    _save_gray(tmp_path / "s2" / "b.png", [[30, 40], [50, 60]])
    _save_gray(tmp_path / "s1" / "z.jpg", [[20, 20], [20, 20]])
    _save_gray(tmp_path / "s1" / "a.pgm", [[0, 10], [20, 30]])

    dataset = load_face_directory(tmp_path)

    assert dataset.class_names == ("s1", "s2")
    assert tuple(path.name for path in dataset.paths) == ("a.pgm", "z.jpg", "b.png")
    np.testing.assert_array_equal(dataset.labels, [0, 0, 1])
    assert dataset.matrix.dtype == np.float64
    assert dataset.matrix.shape == (4, 3)
    assert dataset.image_shape == (2, 2)
    np.testing.assert_allclose(dataset.matrix[:, 0], np.array([0, 10, 20, 30]) / 255.0)
    assert np.all((dataset.matrix >= 0.0) & (dataset.matrix <= 1.0))


def test_resize_uses_height_width_public_order(tmp_path: Path) -> None:
    _save_gray(tmp_path / "subject" / "face.bmp", [[0, 64, 128], [192, 224, 255]])

    dataset = load_face_directory(tmp_path, resize=(3, 2))

    assert dataset.image_shape == (3, 2)
    assert dataset.matrix.shape == (6, 1)


def test_loader_accepts_common_extensions_case_insensitively(tmp_path: Path) -> None:
    extensions = ("pgm", "PNG", "jpg", "JPEG", "bmp", "TIF", "tiff")
    for index, extension in enumerate(extensions):
        _save_gray(tmp_path / "subject" / f"{index}.{extension}", [[index]])

    dataset = load_face_directory(tmp_path)

    assert len(dataset.paths) == len(extensions)


def test_loader_ignores_macos_resource_forks(tmp_path: Path) -> None:
    _save_gray(tmp_path / "subject" / "face.png", [[128]])
    (tmp_path / "subject" / "._face.png").write_bytes(b"not an image")

    dataset = load_face_directory(tmp_path)

    assert tuple(path.name for path in dataset.paths) == ("face.png",)


def test_loader_rejects_images_nested_below_class_directory(tmp_path: Path) -> None:
    nested = tmp_path / "s1" / "session-a" / "face.png"
    _save_gray(nested, [[128]])

    with pytest.raises(ValueError, match=r"s1/session-a/face\.png"):
        load_face_directory(tmp_path)


def test_loader_lists_all_ambiguous_deep_paths(tmp_path: Path) -> None:
    first = tmp_path / "branch-a" / "subject" / "face.png"
    second = tmp_path / "branch-b" / "subject" / "face.png"
    _save_gray(first, [[1]])
    _save_gray(second, [[2]])

    with pytest.raises(ValueError) as error:
        load_face_directory(tmp_path)

    message = str(error.value)
    assert "branch-a/subject/face.png" in message
    assert "branch-b/subject/face.png" in message


def test_face_dataset_validates_shape_and_is_frozen() -> None:
    dataset = FaceDataset(
        matrix=np.zeros((4, 2), dtype=np.float64),
        labels=np.array([0, 1], dtype=np.int64),
        image_shape=(2, 2),
        paths=(Path("a.png"), Path("b.png")),
        class_names=("a", "b"),
    )

    with pytest.raises(FrozenInstanceError):
        dataset.image_shape = (1, 4)  # type: ignore[misc]
    with pytest.raises(ValueError, match="samples"):
        FaceDataset(
            matrix=np.zeros((4, 2), dtype=np.float64),
            labels=np.array([0], dtype=np.int64),
            image_shape=(2, 2),
            paths=(Path("a.png"), Path("b.png")),
            class_names=("a", "b"),
        )


def test_face_dataset_copies_arrays_and_makes_them_read_only() -> None:
    matrix = np.zeros((4, 1), dtype=np.float64)
    labels = np.array([0], dtype=np.int64)

    dataset = FaceDataset(matrix, labels, (2, 2), (Path("a.png"),), ("a",))

    assert not dataset.matrix.flags.writeable
    assert not dataset.labels.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        dataset.matrix[0, 0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        dataset.labels[0] = 1
    matrix[0, 0] = 0.5
    labels[0] = 7
    assert dataset.matrix[0, 0] == 0.0
    assert dataset.labels[0] == 0


@pytest.mark.parametrize(
    ("matrix", "labels", "image_shape", "match"),
    [
        (np.zeros((4, 1), dtype=np.float32), np.array([0]), (2, 2), "float64"),
        (np.zeros((4, 1), dtype=np.float64), np.array([0.0]), (2, 2), "integer"),
        (np.zeros((4, 1), dtype=np.float64), np.array([0]), (1, 3), "pixels"),
    ],
)
def test_face_dataset_rejects_invalid_metadata(
    matrix: np.ndarray, labels: np.ndarray, image_shape: tuple[int, int], match: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        FaceDataset(matrix, labels, image_shape, (Path("a.png"),), ("a",))


def test_loader_rejects_inconsistent_sizes_without_resize(tmp_path: Path) -> None:
    _save_gray(tmp_path / "subject" / "small.png", [[0]])
    _save_gray(tmp_path / "subject" / "large.png", [[0, 1]])

    with pytest.raises(ValueError, match="inconsistent image sizes"):
        load_face_directory(tmp_path)


@pytest.mark.parametrize("resize", [(0, 2), (2, -1), (2,), (2.0, 3)])
def test_loader_rejects_invalid_resize(tmp_path: Path, resize: object) -> None:
    _save_gray(tmp_path / "subject" / "face.png", [[0]])

    with pytest.raises((TypeError, ValueError), match="resize"):
        load_face_directory(tmp_path, resize=resize)  # type: ignore[arg-type]


def test_loader_reports_missing_non_directory_and_empty_roots(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_face_directory(tmp_path / "missing")
    file_root = tmp_path / "file"
    file_root.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError, match="not a directory"):
        load_face_directory(file_root)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="empty"):
        load_face_directory(empty)


def test_loader_distinguishes_unsupported_and_corrupt_images(tmp_path: Path) -> None:
    unsupported = tmp_path / "unsupported"
    (unsupported / "subject").mkdir(parents=True)
    (unsupported / "subject" / "notes.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="no supported image files"):
        load_face_directory(unsupported)

    corrupt = tmp_path / "corrupt"
    (corrupt / "subject").mkdir(parents=True)
    (corrupt / "subject" / "bad.png").write_bytes(b"broken")
    with pytest.raises(ValueError, match=r"corrupt.*bad\.png"):
        load_face_directory(corrupt)


def test_data_api_is_exported_from_package() -> None:
    from robust_nmf import FaceDataset as ExportedDataset
    from robust_nmf import load_face_directory as exported_loader

    assert ExportedDataset is FaceDataset
    assert exported_loader is load_face_directory

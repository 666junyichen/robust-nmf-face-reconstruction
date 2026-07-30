# Local face datasets

Raw dataset files are intentionally excluded from version control. Obtain each dataset from its official source, review its current terms, and keep your own provenance and integrity records. You are responsible for ensuring that your use and redistribution comply with the applicable dataset license or access conditions.

Official source pages:

- [The Database of Faces (formerly ORL)](https://cam-orl.co.uk/facedatabase.html/)
- [Extended Yale Face Database B (B+)](https://vision.ucsd.edu/datasets/extended-yale-face-database-b-b)

The package's MIT license covers newly organized source code only. It does not cover either dataset, cited works, or the anonymized/redacted team technical report.

## Required layout

`load_face_directory` enforces the exact `root/class/image` structure. Place no supported image files directly at the dataset root or more than one directory below it.

```text
data/
├── ORL/
│   ├── subject_01/
│   │   ├── image_01.pgm
│   │   └── image_02.pgm
│   └── subject_02/
│       └── image_01.pgm
└── CroppedYaleB/
    ├── subject_01/
    │   ├── image_01.pgm
    │   └── image_02.pgm
    └── subject_02/
        └── image_01.pgm
```

Directory names become deterministic class labels. Supported extensions are `.pgm`, `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, and `.tiff` (case-insensitive). Hidden macOS resource-fork files beginning with `._` are ignored.

## Loading

Run from an activated environment with the package installed:

```python
from robust_nmf.data import load_face_directory

orl = load_face_directory("data/ORL", resize=(30, 37))
yaleb = load_face_directory("data/CroppedYaleB", resize=(42, 48))

print(orl.matrix.shape, orl.labels.shape, orl.image_shape)
print(yaleb.matrix.shape, yaleb.labels.shape, yaleb.image_shape)
```

`resize` is always `(height, width)`, not `(width, height)`. Images are converted to grayscale, normalized to `[0, 1]`, flattened in row-major order, and stored as matrix columns. Without `resize`, all images under one dataset root must already have the same dimensions.

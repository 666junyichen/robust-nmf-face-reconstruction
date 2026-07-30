[English](README.md) | [简体中文](README.zh-CN.md)

# Robust NMF Face Reconstruction

An AI/ML and data-science portfolio project for studying low-rank face reconstruction under salt-and-pepper corruption. The repository turns a completed team experiment into a testable Python package with deterministic data loading, a validated corruption model, from-scratch L2-NMF, an L2,1-oriented robust factorization with an explicit column-sparse residual, clustering evaluation, and reproducible visualizations.

The experimental workflow targets the ORL and Extended YaleB face datasets. Reconstruction is measured with relative reconstruction error (RRE); coefficient representations are evaluated with clustering accuracy and normalized mutual information (NMI).

## Key results

The values below are three-decimal, rounded aggregates of the means in [`summary.csv`](results/metrics/summary.csv), averaged across the six stored corruption/salt-ratio settings for each dataset and method. They come from completed historical experiments; neither the smoke test nor the current refactor reproduced them.

<!-- aggregate-results:start -->
| Dataset | Method | RRE ↓ | Accuracy ↑ | NMI ↑ |
|---|---:|---:|---:|---:|
| ORL | L2-NMF | 0.407 | 0.185 | 0.357 |
| ORL | L21-NMF | 0.276 | 0.364 | 0.557 |
| Extended YaleB | L2-NMF | 0.670 | 0.178 | 0.304 |
| Extended YaleB | L21-NMF | 0.280 | 0.155 | 0.196 |
<!-- aggregate-results:end -->

Across every stored setting, L21-NMF has lower RRE than L2-NMF. It also has an ORL clustering advantage in both accuracy and NMI. Extended YaleB clustering is mixed: the accuracy winner changes across settings, while L2-NMF has higher NMI throughout the stored grid.

These comparisons require caution. The historical clean-data protocol was asymmetric: L2,1-NMF refit coefficients on clean data while holding the learned basis fixed, whereas L2-NMF evaluated factors learned from noisy data without clean-data refitting. The archived results therefore do not isolate the loss function under a perfectly symmetric protocol. See the [`PROVENANCE.md`](results/metrics/PROVENANCE.md) record for the source digest, table mapping, precision limit, and verification procedure.

![RRE comparison for L2-NMF and L21-NMF across ORL and Extended YaleB corruption settings](results/figures/rre_comparison.png)

![Accuracy and NMI comparison for L2-NMF and L21-NMF across ORL and Extended YaleB corruption settings](results/figures/clustering_comparison.png)

## Team Project and My Contributions

This was a four-person team project. Identities are intentionally omitted.

My contributions were:

- Implemented and validated the salt-and-pepper noise generator.
- Implemented the L2-NMF baseline and checked its objective and update rules against the literature.
- Researched and organized references supporting the background and motivation.
- Contributed the theoretical framing and explanation of the compared methods.

The robust L2,1-oriented implementation and the project as a whole are team outcomes; I do not present either as solely my work.

## Architecture

| Module | Responsibility |
|---|---|
| `robust_nmf.data` | Strict `root/class/image` loading, grayscale conversion, resizing, normalization, and deterministic labels |
| `robust_nmf.noise` | Seeded salt-and-pepper corruption with exact unique-position selection |
| `robust_nmf.nmf` | From-scratch L2-NMF and robust residual-based factorization |
| `robust_nmf.metrics` | Scale-safe RRE plus Hungarian-aligned clustering accuracy and NMI |
| `robust_nmf.visualization` | Stored-summary validation and comparison plots |
| `scripts/` | Synthetic smoke experiment, figure generation, report redaction, and provenance verification |

```text
.
├── data/                         # Local datasets (not versioned)
├── docs/
│   └── robust_nmf_technical_report.pdf
├── notebooks/
│   └── robust_nmf_experiments.ipynb
├── results/
│   ├── figures/                  # Curated aggregate plots
│   └── metrics/                  # Historical summary and provenance
├── scripts/                      # Smoke, plotting, redaction, verification
├── src/robust_nmf/               # Package source
└── tests/                        # Unit, integration, and documentation checks
```

## Quickstart

Python 3.10 or newer is required.

```console
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Activate on Linux or macOS:

```bash
source .venv/bin/activate
```

Install the package and development/report extras, then verify it:

```console
python -m pip install -e ".[dev,report]"
python -m pytest
python scripts/smoke_experiment.py
python scripts/generate_result_figures.py
```

The smoke experiment uses a small deterministic synthetic matrix. It checks finite RRE and nonnegative factors; it is not a benchmark and does not reproduce the stored face-dataset results.

## Data layout

Raw images are intentionally excluded. Follow the licensing and provenance guidance in [`data/README.md`](data/README.md) and place files in the exact loader layout:

```text
data/
├── ORL/
│   ├── subject_01/
│   │   ├── image_01.pgm
│   │   └── ...
│   └── ...
└── CroppedYaleB/
    ├── subject_01/
    │   ├── image_01.pgm
    │   └── ...
    └── ...
```

Every supported image must be exactly one class directory below its dataset root. Supported extensions are `.pgm`, `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, and `.tiff`.

## Notebook and report

Install a notebook UI and launch from the repository root:

```console
python -m pip install jupyterlab
python -m jupyter lab notebooks/robust_nmf_experiments.ipynb
```

The notebook separates optional local-data reruns from visualization of stored aggregate evidence. The accompanying [`robust_nmf_technical_report.pdf`](docs/robust_nmf_technical_report.pdf) is an anonymized/redacted team technical report and remains a historical artifact, not package documentation authored solely by me.

## Reproducibility

- Algorithm, corruption, and clustering entry points accept explicit random seeds.
- The loader uses deterministic path and class ordering.
- `summary.csv` is schema-validated before plotting.
- `python scripts/generate_result_figures.py` regenerates both checked-in aggregate plots.
- `python scripts/verify_result_provenance.py --report PATH --csv results/metrics/summary.csv` verifies a locally available archived report against the recorded digest and table values.
- A new local-data run is a new, symmetric evaluation protocol; it is not an exact replay of the archived experiment.

## Limitations

- The face datasets are not distributed here, so full experiments require independently obtained and verified data.
- Stored values retain only the historical report's three-decimal precision and no run-level observations.
- The historical cross-method evaluation used an asymmetric clean-data protocol.
- The current robust solver is an L2,1-oriented residual model; global convergence and equivalence to every formulation named “L2,1-NMF” are not claimed.
- Clustering results depend on representation quality, initialization, and the chosen evaluation protocol.

## Rights

The [MIT License](LICENSE) applies to the newly organized source code in this repository. The datasets, cited works, and anonymized/redacted team technical report have separate owners and terms; the MIT license does not grant rights to those materials.

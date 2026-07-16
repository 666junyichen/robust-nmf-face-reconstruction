"""Run a deterministic, lightweight invariant check for both NMF methods."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robust_nmf.metrics import relative_reconstruction_error
from robust_nmf.nmf import fit_l2_nmf, fit_l21_nmf
from robust_nmf.noise import add_salt_pepper_noise


def main() -> int:
    rng = np.random.default_rng(17)
    clean = rng.random((18, 3)) @ rng.random((3, 14))
    clean /= clean.max()
    noisy = add_salt_pepper_noise(clean, 0.12, 0.5, seed=17)
    fits = {
        "L2-NMF": fit_l2_nmf(noisy, 3, max_iter=35, tol=0.0, seed=17),
        "L21-NMF": fit_l21_nmf(
            noisy, 3, regularization=0.12, max_iter=35, tol=0.0, seed=17
        ),
    }

    for name, result in fits.items():
        error = relative_reconstruction_error(clean, result.reconstruction)
        valid = (
            np.isfinite(error)
            and np.isfinite(result.basis).all()
            and np.isfinite(result.coefficients).all()
            and (result.basis >= 0).all()
            and (result.coefficients >= 0).all()
        )
        print(f"{name}: RRE={error:.6f}")
        if not valid:
            print(f"FAIL: invariant violation in {name}", file=sys.stderr)
            return 1
    print("PASS: finite RRE and nonnegative factors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

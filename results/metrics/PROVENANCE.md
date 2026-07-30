# Experiment result provenance

`summary.csv` is a transcription from the final archived team technical report snapshot. The exact report bytes are identified without retaining a private filename or filesystem location:

```text
SHA-256: f731e87a11f0456adf354b88532bf37f42247f9833f58f68dabc53e29aea7493
```

## Source and mapping

- The ORL rows come from PDF page 9, Table 3.
- The Extended YaleB rows come from PDF page 10, Table 4.
- Each table row maps `(p, r)` to CSV columns `corruption` and `salt_ratio`.
- Report method `L2,1-norm` maps to `L21-NMF`; `L2-norm` maps to `L2-NMF`.
- Within each row, report values `RRE`, `ACC`, and `NMI` map in that order to the corresponding CSV mean and standard-deviation columns.
- The CSV retains the report's rounded precision of three decimal places. It does not claim access to unrounded run-level observations.

These are historical aggregate results. The protocol was asymmetric: L2,1-NMF refit its coefficient matrix on clean data while keeping `W` fixed, whereas L2-NMF evaluated factors learned from noisy data without clean-data refitting. The stored comparison therefore is not a perfectly symmetric rerun protocol.

## Verification

From the repository root, a maintainer who has the archived snapshot can supply its path:

```console
python scripts/verify_result_provenance.py --report PATH --csv results/metrics/summary.csv
```

The verifier hashes the exact report bytes, validates the CSV schema and complete 24-row experiment grid, extracts PDF text, and matches all six stored mean/standard-deviation tokens in every row to Tables 3–4.

# M2 Mask Polarity Audit - 2026-05-15

## Trigger

Manual inspection of `paper/figures/m2_qualitative_mixed/` showed that only a few qualitative panels were visually convincing. A larger stratified candidate set was generated in `paper/figures/m2_qualitative_candidates/`, then checked against the raw annotation masks.

## Key Finding

The dataset contains two incompatible annotation polarities:

- `mosaic_*` samples use black background with white crack pixels.
- Non-mosaic samples use white background with black crack pixels.

The current training/evaluation/visualization path reads masks as `mask > 127`, so it treats:

- mosaic samples: white crack pixels as foreground, which is correct.
- non-mosaic samples: white background as foreground, which is wrong for crack segmentation.

This explains why M2 metrics are extremely high while many qualitative figures look unsuitable for a crack segmentation paper.

## Split Statistics

`white_ratio = mean(annotation > 127)`

| Split | Total | White Majority | Black Majority | Median White Ratio | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| train | 11059 | 8710 | 2349 | 0.978851 | 0.001007 | 0.999039 |
| val | 1591 | 1088 | 503 | 0.974869 | 0.001026 | 0.998306 |
| test | 1594 | 1090 | 504 | 0.975767 | 0.001034 | 0.999039 |

By filename group:

| Split | Group | N | White Majority | Black Majority | Median White Ratio |
|---|---|---:|---:|---:|---:|
| train | mosaic | 2349 | 0 | 2349 | 0.003445 |
| train | non-mosaic | 8710 | 8710 | 0 | 0.983007 |
| val | mosaic | 503 | 0 | 503 | 0.003426 |
| val | non-mosaic | 1088 | 1088 | 0 | 0.983026 |
| test | mosaic | 504 | 0 | 504 | 0.003374 |
| test | non-mosaic | 1090 | 1090 | 0 | 0.983370 |

## Representative Raw Masks

- `data/val/annotations/214_05_01.png`: white background, black crack.
- `data/val/annotations/mosaic_1561.png`: black background, white crack.

## Generated Candidate Artifacts

- Candidate panels: `paper/figures/m2_qualitative_candidates/`
- Candidate masks/overlays/index: `results/visualizations/m2_qualitative_candidates/`
- Contact sheets:
  - `analysis/figures/m2_candidate_contact_sheet_1.png`
  - `analysis/figures/m2_candidate_contact_sheet_2.png`

## Decision

Do not use the current M2 formal metrics or qualitative panels as final paper evidence. The next task must normalize mask polarity across all data-loading and baseline/visualization paths, then retrain/evaluate M2 from scratch.


# M2 polarity-fixed qualitative figure selection (2026-05-16)

## Screening inputs

- Corrected default panels: `paper/figures/m2_qualitative/`
- Corrected mixed panels: `paper/figures/m2_qualitative_mixed/`
- Contact sheet: `analysis/figures/m2_polarityfix_contact_sheet_2026-05-16.png`
- Selected figure copy: `paper/figures/m2_selected/`
- Model checkpoint: `checkpoints/m2_lora_polarityfix/best_model.pth`

## Selected paper-body candidates

| Stem | Source panel | GT ratio | LoRA ratio | LoRA IoU | Reason |
| --- | --- | ---: | ---: | ---: | --- |
| `214_01_01` | `paper/figures/m2_qualitative_mixed/214_01_01_comparison.png` | 0.095681 | 0.099449 | 0.845353 | Large continuous crack; SAM+LoRA follows GT closely while SAM-Box over-segments heavily. |
| `1615_07_01` | `paper/figures/m2_qualitative_mixed/1615_07_01_comparison.png` | 0.025131 | 0.027657 | 0.527162 | Medium branching crack; useful for showing topology recovery and AMG failure. |
| `347_01_01` | `paper/figures/m2_qualitative/347_01_01_comparison.png` | 0.002983 | 0.008972 | 0.161816 | Thin low-contrast crack; LoRA keeps the main crack continuity, Box-Oracle has obvious false positives. |
| `1220_04_01` | `paper/figures/m2_qualitative/1220_04_01_comparison.png` | 0.002711 | 0.003498 | 0.119690 | Fine horizontal crack with low contrast; useful as a difficult small-foreground example. |

## Reserve or supplementary candidates

| Stem | Status | Reason |
| --- | --- | --- |
| `271_06_01` | Reserve | Natural branching crack, but SAM+LoRA misses enough branches that it is better used as a failure/limitation example. |
| `mosaic_0777` | Supplement only | Shows mixed polarity/mosaic behavior, but tile boundaries make it less suitable for the main paper figure. |
| `mosaic_2569` | Supplement only | Very small foreground and visible mosaic tiling; useful for audit, not ideal for reader-facing comparison. |
| `mosaic_3870` | Supplement only | Good IoU on a tiny foreground, but the visible blank/tile structure can distract from method comparison. |

## Decision

Use the four selected real-image panels for the M2 qualitative comparison in the paper body. Keep mosaic panels as supplementary evidence for polarity robustness or data-audit discussion, not as the primary qualitative figure.

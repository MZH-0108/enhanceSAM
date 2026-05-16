# M2 corrected result archive

This directory is the consolidated paper-facing archive for the polarity-fixed M2 mainline result.

## Use these artifacts for the paper

- Quantitative table: `tables/m2_formal_comparison.csv`
- Main qualitative panels: `figures/`
- Quantitative report: `analysis/m2_formal_comparison_2026-05-16.md`
- Qualitative selection note: `analysis/m2_qualitative_selection_2026-05-16.md`
- Result wording draft: `analysis/m2_result_wording_2026-05-16.md`
- Screening contact sheet: `analysis/m2_polarityfix_contact_sheet_2026-05-16.png`

## Final mainline checkpoint

- `checkpoints/m2_lora_polarityfix/best_model.pth`
- Config: `configs/m2_lora_polarityfix_config.yaml`

## Important caveat

Do not use the older `checkpoints/m2_lora/` results in final paper tables or figures. Those were produced before annotation polarity was normalized and are retained only as diagnostic history.

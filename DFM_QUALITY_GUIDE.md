# DFM Quality Guide

## Why blank or weak DFM happens

- Training stops too early (low iteration count).
- Faceset after filtering is too small or too noisy.
- Source and destination distribution mismatch (pose/light/expression).
- Over-aggressive defaults on low-duration jobs.

## Quality gates added in this repo

- Minimum kept faces for source (`AUTOTRAIN_MIN_SRC_FACES`)
- Minimum kept faces for destination (`AUTOTRAIN_MIN_DST_FACES`)
- Minimum training iterations (`AUTOTRAIN_MIN_ITERS`)
- Export only after all checks pass

Reports are generated at:

- `reports/quality_report.json`
- `reports/quality_report.md`

## Suggested production defaults

- `max_hours`: 6-12
- `plateau_hours`: 2-4
- `AUTOTRAIN_MIN_ITERS`: 2000 (increase for hard datasets)
- `AUTOTRAIN_MIN_SRC_FACES`: 300+
- `AUTOTRAIN_MIN_DST_FACES`: 300+

## Operational checklist

1. Validate both videos are readable and long enough.
2. Check filter reports (`src_filter_report.json`, `dst_filter_report.json`).
3. Confirm iterations and best loss in `train_report.json`.
4. Only consume `.dfm` from runs where `ready_for_export=true`.

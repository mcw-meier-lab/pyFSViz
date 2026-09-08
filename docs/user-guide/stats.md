---
title: Stats and outliers
---

# Stats and outliers

The group HTML report is a wrapper around helpers in `pyfsviz.stats`. Use them
when you want CSVs, outlier lists, or plots without generating HTML.

FreeSurfer must be initialized: `get_stats()` runs `asegstats2table` and
`aparcstats2table`. See [Prerequisites](prerequisites.md).

## Collect tables

```python
from pyfsviz.stats import get_stats

stats = get_stats(["sub-001", "sub-002"], "reports/")
# stats["aseg"] -> Path to aseg.csv
# stats["aparc"] -> list of aparc CSV paths, including combined_aparc.csv
```

`SUBJECTS_DIR` must point at the tree that contains those subject folders.
Optional `measures` (default `area`, `volume`, `thickness`) and `hemis`
(default `lh`, `rh`) control which aparc tables are written.

## Outliers

`check_metrics()` flags values more than `sd_threshold` standard deviations
from the cohort mean, per region:

```python
from pyfsviz.stats import check_metrics, summarize_outlier_subjects

from pathlib import Path

files = list(Path("reports").glob("*.csv"))
quality = check_metrics(files, sd_threshold=3.0)
outliers = summarize_outlier_subjects(quality)
```

Each outlier entry has `subject_id`, `outlier_count`, and `findings` (metric
and region with the raw value).

## Group comparisons

```python
from pyfsviz.stats import gen_group_comparison_plots

groups = {
    "control": ["sub-001", "sub-002"],
    "patient": ["sub-101", "sub-102"],
}
figures = gen_group_comparison_plots(files, groups)
```

These are Plotly box plots (one per metric region) with a point per subject.
The group HTML report embeds the same figures in one tab per stats table
(aseg, LH Area, RH Thickness, …).

## Distribution plots

```python
from pyfsviz.stats import gen_metric_plots

plots = gen_metric_plots(files)
```

These are Plotly figures (the same family embedded in the group report).
One figure is created per region in each stats table (`aseg`, `lh_area_aparc`,
and so on).

For the HTML wrapper, see [Group reports](group-reports.md).

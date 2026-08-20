"""Custom nipype interfaces for FreeSurfer stats commands."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from nipype.interfaces.base import (
    File,
    InputMultiObject,
    TraitedSpec,
    traits,
)
from nipype.interfaces.freesurfer.base import FSCommand, FSTraitedSpec

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover
    scipy_stats = None

_MIN_GROUP_SAMPLES = 2
_MIN_GROUPS = 2
_SYNTHSEG_VOL_CSV = Path("stats") / "synthseg.vol.csv"
_SYNTHSEG_TIV_COLUMN = "total intracranial"
_APARC_REPEAT_COLUMNS = ("BrainSegVolNotVent", "eTIV")
_logger = logging.getLogger(__name__)


def _normalize_subject_id(subject_id: str) -> str:
    if "/" in subject_id:
        return subject_id.rsplit("/", maxsplit=1)[-1]
    return str(subject_id)


def _synthseg_tiv_column(columns: pd.Index) -> str | None:
    for column in columns:
        if str(column).strip().lower() == _SYNTHSEG_TIV_COLUMN:
            return str(column)
    return None


def _read_synthseg_tiv(subject: str, subjects_dir: str | Path | None = None) -> float | None:
    """Return SynthSeg total intracranial volume for a subject, if available."""
    base = Path(subjects_dir) if subjects_dir is not None else Path(os.environ.get("SUBJECTS_DIR", "."))
    csv_path = base / _normalize_subject_id(subject) / _SYNTHSEG_VOL_CSV
    if not csv_path.is_file():
        return None

    try:
        df = pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        _logger.warning("Could not read SynthSeg volumes for %s: %s", subject, exc)
        return None

    tiv_col = _synthseg_tiv_column(df.columns)
    if tiv_col is None or df.empty:
        return None

    id_col = df.columns[0]
    if str(id_col).strip().lower() != _SYNTHSEG_TIV_COLUMN:
        subject_key = _normalize_subject_id(subject)
        matches = df[df[id_col].map(lambda value: _normalize_subject_id(str(value)) == subject_key)]
        row = matches.iloc[0] if not matches.empty else df.iloc[0]
    else:
        row = df.iloc[0]

    value = row[tiv_col]
    if pd.isna(value):
        return None
    return float(value)


def _add_synthseg_tiv_to_aseg(
    aseg_file: Path,
    subjects: list[str],
    subjects_dir: str | Path | None = None,
    *,
    column: str = _SYNTHSEG_TIV_COLUMN,
) -> Path:
    """Add SynthSeg total intracranial volume to an aggregated aseg table."""
    df = pd.read_csv(aseg_file)
    if df.empty:
        return aseg_file

    id_col = df.columns[0]
    tiv_by_subject = {_normalize_subject_id(subject): _read_synthseg_tiv(subject, subjects_dir) for subject in subjects}
    mapped = df[id_col].map(lambda sid: tiv_by_subject.get(_normalize_subject_id(str(sid))))

    if column in df.columns:
        df[column] = mapped
    else:
        df.insert(1, column, mapped)

    df.to_csv(aseg_file, index=False)
    if mapped.isna().all():
        _logger.warning("No SynthSeg total intracranial values found for aseg table %s", aseg_file)
    return aseg_file


class AsegStatsInputSpec(FSTraitedSpec):
    """Input specification for asegstats2table command."""

    # asegstats2table --subjects --meas volume --delimiter=comma --skip --tablefile
    subjects = InputMultiObject(
        traits.Str(),
        argstr="%s...",
        desc="subjects to pull stats from",
        mandatory=True,
        position=1,
    )
    meas = traits.Enum("volume", "mean", argstr="--meas %s", desc="measure to output")
    delim = traits.Enum(
        "comma",
        "tab",
        "space",
        "semicolon",
        argstr="--delimiter=%s",
    )
    skip = traits.Bool(argstr="--skip", desc="skip empty files")
    tablefile = File(
        argstr="--tablefile %s",
        exists=False,
        desc="Output file name",
        mandatory=True,
    )
    transpose = traits.Bool(argstr="--transpose", desc="transpose table")
    segs = traits.Bool(argstr="--all-segs", desc="use all segs available")


class AsegStatsOutputSpec(TraitedSpec):
    """Output specification for asegstats2table command."""

    out_table = File(desc="output file")


class AsegStats(FSCommand):
    """Custom nipype interface for FreeSurfer asegstats2table command."""

    _cmd = "asegstats2table --subjects"
    input_spec = AsegStatsInputSpec
    output_spec = AsegStatsOutputSpec

    def run(self, **inputs: Any) -> dict:
        """Run asegstats2table command."""
        return super().run(**inputs)

    def _list_outputs(self) -> dict[str, Path]:
        outputs = self._outputs().get()
        outputs["out_table"] = self.inputs.tablefile
        return outputs


class AparcStatsInputSpec(FSTraitedSpec):
    # aparcstats2table --subjects --skip --delimiter=comma --meas area volume thickness --hemi --tablefile
    """Input specification for aparcstats2table command."""

    subjects = InputMultiObject(
        traits.Str(),
        argstr="%s...",
        mandatory=True,
        desc="subjects to pull aparc stats",
        position=1,
    )
    hemi = traits.Enum(
        "lh",
        "rh",
        argstr="--hemi %s",
        mandatory=True,
        desc="hemisphere to use",
    )
    meas = traits.Enum(
        "area",
        "volume",
        "thickness",
        "thicknessstd",
        "meancurv",
        "gauscurv",
        "foldind",
        "curvind",
        argstr="--meas %s",
        desc="measure",
    )
    delim = traits.Enum(
        "tab",
        "comma",
        "space",
        "semicolon",
        argstr="--delimiter=%s",
        desc="table delimiter",
    )
    parc = traits.Str(argstr="--parc %s", desc="parcellation to use")
    skip = traits.Bool(argstr="--skip", desc="skip empty inputs")
    tablefile = File(
        argstr="--tablefile %s",
        mandatory=True,
        exists=False,
        desc="output file name",
    )
    transpose = traits.Bool(argstr="--transpose", desc="transpose table")


class AparcStatsOutputSpec(TraitedSpec):
    """Output specification for aparcstats2table command."""

    out_table = File(desc="output file")


class AparcStats(FSCommand):
    """Custom nipype interface for FreeSurfer aparcstats2table command."""

    _cmd = "aparcstats2table --subjects"
    input_spec = AparcStatsInputSpec
    output_spec = AparcStatsOutputSpec

    def run(self, **inputs: Any) -> dict:
        """Run aparcstats2table command."""
        return super().run(**inputs)

    def _list_outputs(self) -> dict[str, Path]:
        outputs = self._outputs().get()
        outputs["out_table"] = self.inputs.tablefile
        return outputs


def _get_aseg_stats(
    subjects: list[str],
    tablefile: str,
    meas: str = "volume",
    delim: str = "comma",
    output_dir: str = ".",
    *,
    skip: bool = True,
    segs: bool = True,
) -> Path:
    """Generate aseg table.

    Parameters
    ----------
    subjects : list
        List of subject IDs to use
    tablefile : str
        Name of output file
    meas : str, optional
        Choose from volume, area. By default "volume"
    delim : str, optional
        String delimiter to use, by default "comma"
    skip : bool, optional
        Skip rather than crash if missing data, by default True
    segs : bool, optional
        Use all-segs flag, by default True
    output_dir : str, optional
        Output directory, by default "."

    Returns
    -------
    Path
        Path to output tablefile.
    """
    aseg_cmd = AsegStats(
        subjects=subjects,
        meas=meas,
        delim=delim,
        skip=skip,
        tablefile=Path(output_dir, tablefile),
        segs=segs,
    )
    aseg_cmd.run()
    aseg_file = aseg_cmd._list_outputs()["out_table"]
    aseg_path = Path(output_dir, aseg_file)
    return _add_synthseg_tiv_to_aseg(aseg_path, subjects)


def _combine_aparc_tables(table_files: list[Path]) -> pd.DataFrame:
    """Merge per-hemisphere, per-measure aparc tables into one wide table.

    ``aparcstats2table`` column names already encode hemisphere, region, and
    measure (for example ``lh_bankssts_area``). Metadata columns that are
    repeated in every table are kept once.
    """
    combined: pd.DataFrame | None = None
    for table_file in table_files:
        df = pd.read_csv(table_file)
        df = df.rename(columns={df.columns[0]: "subject_id"})
        df["subject_id"] = df["subject_id"].astype(str).map(_normalize_subject_id)
        if combined is None:
            combined = df
            continue
        overlap = [col for col in df.columns if col in combined.columns and col != "subject_id"]
        combined = combined.merge(df.drop(columns=overlap), on="subject_id", how="outer")

    if combined is None:
        return pd.DataFrame()

    trailing = [col for col in _APARC_REPEAT_COLUMNS if col in combined.columns]
    leading = [col for col in combined.columns if col not in trailing]
    return combined[leading + trailing]


def _get_aparc_stats(
    subjects: list[str],
    tablefile: str,
    *,
    measures: list[str] | None = None,
    hemis: list[str] | None = None,
    delim: str = "comma",
    parc: str = "aparc",
    output_dir: str = ".",
    skip: bool = True,
) -> list[Path]:
    """Generate parcellation stats.

    Parameters
    ----------
    subjects : list
        List of subject IDs
    tablefile : str
        Name of output file
    measures : str, optional
        Choose one of , by default None
    hemis : str, optional
        Choose one of ['lh','rh'], will run both by default.
    delim : str, optional
        String delimiter, by default "comma"
    parc : str, optional
        Parcellation to use, by default "aparc"
    skip : bool, optional
        Skip rather than crash if missing data, by default True
    output_dir : str, optional
        Output directory, by default "."

    Returns
    -------
    list
        List of paths to output files
    """
    if measures is None:
        measures = ["area", "volume", "thickness"]
    if hemis is None:
        hemis = ["lh", "rh"]

    results = []

    for m in measures:
        for h in hemis:
            aparc_cmd = AparcStats(
                subjects=subjects,
                meas=m,
                hemi=h,
                delim=delim,
                skip=skip,
                tablefile=Path(output_dir, f"{h}_{m}_{tablefile}"),
                parc=parc,
            )
            aparc_cmd.run()
            res = aparc_cmd._list_outputs()
            results.append(Path(output_dir, res["out_table"]))

    combined_path = Path(output_dir, f"combined_{tablefile}")
    _combine_aparc_tables(results).to_csv(combined_path, index=False)
    results.append(combined_path)

    return results


def get_stats(
    subjects: list[str],
    output_dir: str,
    measures: list[str] | None = None,
    hemis: list[str] | None = None,
) -> dict[str, Path | list[Path]]:
    """Get aseg and aparc stats from subjects.

    Parameters
    ----------
    subjects : list
        List of subject IDs
    output_dir : str
    measures : list, optional
        List of measures to get, by default None
    hemis : list, optional
        List of hemispheres to get, by default None
    """
    stats: dict[str, Path | list[Path]] = {}
    stats["aseg"] = _get_aseg_stats(subjects, "aseg.csv", output_dir=output_dir)
    stats["aparc"] = _get_aparc_stats(subjects, "aparc.csv", output_dir=output_dir, measures=measures, hemis=hemis)
    return stats


def _subject_group_map(groups: dict[str, list[str]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for group_name, group_subjects in groups.items():
        for subject in group_subjects:
            mapping[_normalize_subject_id(subject)] = group_name
    return mapping


def _load_metrics(stats_files: list[Path]) -> dict[str, pd.DataFrame]:
    metrics: dict[str, pd.DataFrame] = {}
    for file in stats_files:
        if "combined" in file.stem:
            continue
        metrics[file.stem] = pd.read_csv(file)
    return metrics


def _region_columns(data: pd.DataFrame) -> list[str]:
    return [
        col
        for col in data.columns[1:]
        if col not in ["Measure:volume", "lh.aparc.a2009s_thickness", "rh.aparc.a2009s_thickness", "hemi"]
    ]


def _group_values(
    data: pd.DataFrame,
    region: str,
    groups: dict[str, list[str]],
) -> dict[str, np.ndarray]:
    id_col = data.columns[0]
    subject_groups = _subject_group_map(groups)
    grouped: dict[str, list[float]] = {group_name: [] for group_name in groups}

    for _, row in data.iterrows():
        subject_id = _normalize_subject_id(str(row[id_col]))
        group_name = subject_groups.get(subject_id)
        if group_name is None:
            continue
        value = row[region]
        if pd.notna(value):
            grouped[group_name].append(float(value))

    return {group_name: np.array(values, dtype=float) for group_name, values in grouped.items()}


def _compare_two_groups(values_a: np.ndarray, values_b: np.ndarray, alpha: float) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_a": len(values_a),
        "n_b": len(values_b),
        "mean_a": float(values_a.mean()) if len(values_a) else None,
        "mean_b": float(values_b.mean()) if len(values_b) else None,
        "std_a": float(values_a.std(ddof=1)) if len(values_a) > 1 else None,
        "std_b": float(values_b.std(ddof=1)) if len(values_b) > 1 else None,
        "p_value": None,
        "significant": False,
        "test": None,
    }
    if len(values_a) < _MIN_GROUP_SAMPLES or len(values_b) < _MIN_GROUP_SAMPLES:
        summary["message"] = "Insufficient data for statistical comparison"
        return summary

    if scipy_stats is None:
        summary["message"] = "Install scipy for p-values between groups"
        return summary

    result = scipy_stats.ttest_ind(values_a, values_b, equal_var=False)
    summary["p_value"] = float(result.pvalue)
    summary["test"] = "welch_ttest"
    summary["significant"] = result.pvalue < alpha
    summary["message"] = f"p={result.pvalue:.4f} ({'significant' if result.pvalue < alpha else 'not significant'})"
    return summary


def summarize_outlier_subjects(quality_summary: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Aggregate outlier findings by subject for quick reference.

    Parameters
    ----------
    quality_summary
        Output from :func:`check_metrics`.

    Returns
    -------
    list[dict]
        Sorted list of subjects with outlier counts and findings.
    """
    subject_findings: dict[str, list[str]] = defaultdict(list)

    for metric_name, metric_data in quality_summary.items():
        for region, result in metric_data.items():
            if result.get("status") != "outliers_detected":
                continue
            for outlier in result.get("outlier_subjects", []):
                subject_id = _normalize_subject_id(str(outlier["subject_id"]))
                finding = f"{metric_name}/{region}: {float(outlier['value']):.2f}"
                if finding not in subject_findings[subject_id]:
                    subject_findings[subject_id].append(finding)

    return sorted(
        [
            {
                "subject_id": subject_id,
                "outlier_count": len(findings),
                "findings": findings,
            }
            for subject_id, findings in subject_findings.items()
        ],
        key=lambda item: (-item["outlier_count"], item["subject_id"]),
    )


def compare_group_metrics(
    stats_files: list[Path],
    groups: dict[str, list[str]],
    *,
    alpha: float = 0.05,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compare FreeSurfer metrics between named subject groups.

    Parameters
    ----------
    stats_files
        Paths to stats CSV files from :func:`get_stats`.
    groups
        Mapping of group name to subject IDs, e.g. ``{"control": [...], "patient": [...]}``.
    alpha
        Significance threshold for two-group comparisons.

    Returns
    -------
    dict
        Nested comparison results keyed by metric file and brain region.
    """
    if len(groups) < _MIN_GROUPS:
        msg = "At least two groups are required for comparison"
        raise ValueError(msg)

    comparison: dict[str, dict[str, dict[str, Any]]] = {}
    metrics = _load_metrics(stats_files)
    group_names = list(groups)

    for metric_name, data in metrics.items():
        comparison[metric_name] = {}
        for region in _region_columns(data):
            grouped_values = _group_values(data, region, groups)
            region_summary: dict[str, Any] = {
                group_name: {
                    "n": len(grouped_values[group_name]),
                    "mean": float(grouped_values[group_name].mean()) if len(grouped_values[group_name]) else None,
                    "std": float(grouped_values[group_name].std(ddof=1))
                    if len(grouped_values[group_name]) > 1
                    else None,
                }
                for group_name in group_names
            }

            if len(group_names) == _MIN_GROUPS:
                region_summary["comparison"] = _compare_two_groups(
                    grouped_values[group_names[0]],
                    grouped_values[group_names[1]],
                    alpha,
                )
            elif scipy_stats is None:
                region_summary["comparison"] = {
                    "message": "Install scipy for multi-group ANOVA",
                    "test": None,
                    "p_value": None,
                    "significant": False,
                }
            else:
                samples = [
                    grouped_values[name] for name in group_names if len(grouped_values[name]) >= _MIN_GROUP_SAMPLES
                ]
                if len(samples) >= _MIN_GROUPS:
                    result = scipy_stats.f_oneway(*samples)
                    region_summary["comparison"] = {
                        "test": "one_way_anova",
                        "p_value": float(result.pvalue),
                        "significant": result.pvalue < alpha,
                        "message": (
                            f"p={result.pvalue:.4f} ({'significant' if result.pvalue < alpha else 'not significant'})"
                        ),
                    }
                else:
                    region_summary["comparison"] = {
                        "message": "Insufficient data for multi-group comparison",
                        "test": None,
                        "p_value": None,
                        "significant": False,
                    }

            comparison[metric_name][region] = region_summary

    return comparison


def gen_group_comparison_plots(stats_files: list[Path], groups: dict[str, list[str]]) -> list[go.Figure]:
    """Generate Plotly box plots comparing metrics across groups.

    Parameters
    ----------
    stats_files
        Paths to stats CSV files from :func:`get_stats`.
    groups
        Mapping of group name to subject IDs.

    Returns
    -------
    list[go.Figure]
        Plotly figures with one plot per metric region.
    """
    plots: list[go.Figure] = []
    metrics = _load_metrics(stats_files)
    group_names = list(groups)

    for metric_name, data in metrics.items():
        id_col = data.columns[0]
        subject_groups = _subject_group_map(groups)

        for region in _region_columns(data):
            plot_rows = []
            for _, row in data.iterrows():
                subject_id = _normalize_subject_id(str(row[id_col]))
                group_name = subject_groups.get(subject_id)
                value = row[region]
                if group_name is None or pd.isna(value):
                    continue
                plot_rows.append(
                    {
                        "group": group_name,
                        "value": float(value),
                        "subject_id": subject_id,
                    },
                )

            if not plot_rows:
                continue

            plot_data = pd.DataFrame(plot_rows)
            fig = go.Figure()
            for group_name in group_names:
                group_data = plot_data[plot_data["group"] == group_name]
                if group_data.empty:
                    continue
                fig.add_trace(
                    go.Box(
                        y=group_data["value"],
                        name=group_name,
                        text=group_data["subject_id"],
                        boxpoints="all",
                    ),
                )

            fig.update_layout(
                boxmode="group",
                title={"text": f"{metric_name}: {region}"},
                yaxis={"title": {"text": region}},
                xaxis={"title": {"text": "Group"}},
            )
            plots.append(fig)

    return plots


def check_metrics(stats_files: list[Path], sd_threshold: float = 3.0) -> dict:
    """Check metrics from stats files.

    Parameters
    ----------
    stats_files : list
        List of paths to stats files
    sd_threshold : float, optional
        Standard deviation threshold, by default 3.0
    """
    metrics: dict[str, pd.DataFrame] = {}
    metric_summary: dict[str, dict[str, dict[str, Any]]] = {}
    for file in stats_files:
        if "combined" in file.stem:
            continue
        df = pd.read_csv(file)
        metrics[file.stem] = df

    for metric, data in metrics.items():
        # Initialize metric_summary for this metric
        metric_summary[metric] = {}

        # Get column names - skip first column (subject_id) and last few columns (typically metadata)
        region_cols = [
            col
            for col in data.columns[1:]
            if col not in ["Measure:volume", "lh.aparc.a2009s_thickness", "rh.aparc.a2009s_thickness"]
        ]
        id_col = data.columns[0]

        for region in region_cols:
            values = data[region].dropna()
            if len(values) == 0:
                metric_summary[metric][region] = {
                    "status": "no_data",
                    "message": "No data available",
                }
            else:
                values = values.astype(float)
                mean = values.mean()
                std_val = values.std()
                upper_bound = mean + sd_threshold * std_val
                lower_bound = mean - sd_threshold * std_val
                outliers = values[(values > upper_bound) | (values < lower_bound)]

                if len(outliers) > 0:
                    outlier_percentage = (len(outliers) / len(values)) * 100
                    outlier_subjects = []
                    for outlier_val in outliers:
                        outlier_rows = data[data[region] == outlier_val]
                        for _, row in outlier_rows.iterrows():
                            subject_id = row[id_col]
                            outlier_subjects.append(
                                {
                                    "subject_id": str(subject_id),
                                    "value": float(outlier_val),
                                },
                            )

                    unique_outliers = []
                    seen = set()
                    for outlier in outlier_subjects:
                        key = (outlier["subject_id"], outlier["value"])
                        if key not in seen:
                            unique_outliers.append(outlier)
                            seen.add(key)

                    metric_summary[metric][region] = {
                        "status": "outliers_detected",
                        "message": f"Found {len(outliers)} outliers ({outlier_percentage:.1f}%) beyond {sd_threshold} SD",
                        "outlier_count": len(outliers),
                        "outlier_percentage": outlier_percentage,
                        "outlier_subjects": unique_outliers,
                        "mean": mean,
                        "std": std_val,
                        "sd_threshold": sd_threshold,
                        "upper_bound": upper_bound,
                        "lower_bound": lower_bound,
                    }
                else:
                    metric_summary[metric][region] = {
                        "status": "passed",
                        "message": f"No outliers detected (mean: {mean:.2f}, ±{sd_threshold} SD: {lower_bound:.2f} to {upper_bound:.2f})",
                        "outlier_count": 0,
                        "outlier_percentage": 0.0,
                        "outlier_subjects": [],
                        "mean": mean,
                        "std": std_val,
                        "sd_threshold": sd_threshold,
                        "upper_bound": upper_bound,
                        "lower_bound": lower_bound,
                    }
    return metric_summary


def gen_metric_plots(stats_files: list[Path]) -> list:
    """Generate plots from FreeSurfer stats files.

    Parameters
    ----------
    stats_files: list
        List of paths to stats files

    Returns
    -------
    list
        List of plotly figure objects
    """
    plots = []
    metrics = {}
    for file in stats_files:
        if "lh" in file.stem or "rh" in file.stem:
            continue
        df = pd.read_csv(file)
        metrics[file.stem] = df

    for metric, data in metrics.items():
        idx_col = data.columns[0]
        if "hemi" in data.columns:
            for c in data.columns[1:]:
                fig = go.Figure()
                fig.add_trace(
                    go.Box(
                        y=data[data["hemi"] == "lh"][c],
                        boxpoints="suspectedoutliers",
                        marker={
                            "outliercolor": "rgb(0,0,0)",
                            "line": {"outlierwidth": 1, "outliercolor": "rgb(0,0,0)"},
                        },
                        name="lh",
                        text=data["subject_id"],
                    ),
                )
                fig.add_trace(
                    go.Box(
                        y=data[data["hemi"] == "rh"][c],
                        boxpoints="suspectedoutliers",
                        marker={
                            "outliercolor": "rgb(0,0,0)",
                            "line": {"outlierwidth": 1, "outliercolor": "rgb(0,0,0)"},
                        },
                        name="rh",
                        text=data["subject_id"],
                    ),
                )
                fig.update_layout(
                    boxmode="group",
                    yaxis={"title": {"text": c}},
                    xaxis={"title": {"text": "hemisphere"}},
                    title={"text": metric},
                )
                plots.append(fig)
        elif any("Left-" in c for c in data.columns):
            region_groups: dict[str, dict[str, str]] = {}
            for region in data.columns[1:]:  # Skip subject_id column
                # Extract base region name (remove hemisphere prefix if present)
                if region.startswith("Left-"):
                    base_region = region[5:]  # Remove 'Left-' prefix
                    hemisphere = "Left"
                elif region.startswith("Right-"):
                    base_region = region[6:]  # Remove 'Right-' prefix
                    hemisphere = "Right"
                elif region.startswith(("lh", "rh")):
                    base_region = region[2:]  # Remove 'lh' or 'rh' prefix
                    hemisphere = "Left" if region.startswith("lh") else "Right"
                else:
                    # No hemisphere prefix, treat as bilateral
                    base_region = region
                    hemisphere = "Bilateral"

                if base_region not in region_groups:
                    region_groups[base_region] = {}
                region_groups[base_region][hemisphere] = region

            for base_region, hemispheres in region_groups.items():
                fig = go.Figure()
                if len(hemispheres) > 1 and "Bilateral" not in hemispheres:
                    # Multiple hemispheres found, create combined plot
                    combined_data = []
                    for hemisphere, region_col in hemispheres.items():
                        region_data = data[[idx_col, region_col]].copy()
                        region_data = region_data.rename(columns={region_col: "value"})
                        region_data["hemisphere"] = hemisphere
                        combined_data.append(region_data)

                    if combined_data:
                        # Concatenate data from both hemispheres
                        plot_data = pd.concat(combined_data, ignore_index=True)

                        # Create box plot comparing hemispheres
                        fig.add_trace(
                            go.Box(
                                y=plot_data[plot_data["hemisphere"] == "Left"]["value"],
                                boxpoints="suspectedoutliers",
                                text=plot_data[idx_col],
                                name="left",
                                marker={
                                    "outliercolor": "rgb(0,0,0)",
                                    "line": {"outlierwidth": 1, "outliercolor": "rgb(0,0,0)"},
                                },
                            ),
                        )
                        fig.add_trace(
                            go.Box(
                                y=plot_data[plot_data["hemisphere"] == "Right"]["value"],
                                boxpoints="suspectedoutliers",
                                text=plot_data[idx_col],
                                name="right",
                                marker={
                                    "outliercolor": "rgb(0,0,0)",
                                    "line": {"outlierwidth": 1, "outliercolor": "rgb(0,0,0)"},
                                },
                            ),
                        )
                        fig.update_layout(
                            boxmode="group",
                            yaxis={"title": {"text": base_region}},
                            xaxis={"title": {"text": "hemisphere"}},
                        )
                        plots.append(fig)
                else:
                    region_col = next(iter(hemispheres.values()))
                    fig.add_trace(
                        go.Box(
                            y=data[region_col],
                            boxpoints="suspectedoutliers",
                            text=data[idx_col],
                            name=base_region,
                            marker={
                                "outliercolor": "rgb(0,0,0)",
                                "line": {"outlierwidth": 1, "outliercolor": "rgb(0,0,0)"},
                            },
                        ),
                    )
                    fig.update_layout(yaxis={"title": {"text": base_region}})
                    plots.append(fig)
        else:
            # Handle aseg data (no hemisphere prefix)
            for region in data.columns[1:]:  # Skip subject_id column
                fig = go.Figure()
                fig.add_trace(
                    go.Box(
                        y=data[region],
                        boxpoints="suspectedoutliers",
                        text=data[idx_col],  # subject_id column
                        name=region,
                        marker={
                            "outliercolor": "rgb(0,0,0)",
                            "line": {"outlierwidth": 1, "outliercolor": "rgb(0,0,0)"},
                        },
                    ),
                )
                fig.update_layout(
                    yaxis={"title": {"text": region}},
                    title={"text": metric},
                )
                plots.append(fig)

    return plots

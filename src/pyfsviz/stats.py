"""Custom nipype interfaces for FreeSurfer stats commands."""

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from nipype.interfaces.base import (
    File,
    InputMultiObject,
    TraitedSpec,
    traits,
)
from nipype.interfaces.freesurfer.base import FSCommand, FSTraitedSpec


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
    subjects: list[str] | pd.Series,
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
    subjects : list or pandas.Series
        List of subject IDs to use. If a pandas Series is provided, it will be
        converted to a list of strings.
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
    # Convert pandas Series to list if needed
    if isinstance(subjects, pd.Series):
        subjects = subjects.tolist()
    # Ensure all elements are strings
    subjects = [str(s) for s in subjects]

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
    return Path(output_dir, aseg_file)


def _get_aparc_stats(
    subjects: list[str] | pd.Series,
    tablefile: str,
    measures: list[str] | None = None,
    hemis: list[str] | None = None,
    delim: str = "comma",
    parc: str = "aparc",
    output_dir: str = ".",
    *,
    skip: bool = True,
) -> list[Path]:
    """Generate parcellation stats.

    Parameters
    ----------
    subjects : list or pandas.Series
        List of subject IDs. If a pandas Series is provided, it will be
        converted to a list of strings.
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
    # Convert pandas Series to list if needed
    if isinstance(subjects, pd.Series):
        subjects = subjects.tolist()
    # Ensure all elements are strings
    subjects = [str(s) for s in subjects]

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

        combined_df = pd.DataFrame()
        for file in results:
            df = pd.read_csv(str(file))
            label = df.columns[0]
            df.rename(columns={label: "subject_id"}, inplace=True)

            df["hemi"] = file.stem.split("_")[0]
            cols = df.columns.tolist()[1:-3]
            for c in cols:
                col_name = c.split("_")[1]
                df.rename(columns={c: col_name}, inplace=True)
            combined_df = pd.concat([combined_df, df])

        subjects = combined_df["subject_id"]
        new_subjects = []
        for subj in subjects:
            if "/" in subj:
                new_subjects.append(subj.split("/")[-1])
            else:
                new_subjects.append(subj)
        combined_df["subject_id"] = new_subjects
        combined_df.to_csv(Path(output_dir, f"combined_{tablefile}"), index=False)
        results.append(Path(output_dir, f"combined_{tablefile}"))

    return results


def get_stats(
    subjects: list[str] | pd.Series,
    output_dir: str,
    measures: list[str] | None = None,
    hemis: list[str] | None = None,
) -> dict[str, Path | list[Path]]:
    """Get aseg and aparc stats from subjects.

    Parameters
    ----------
    subjects : list or pandas.Series
        List of subject IDs. If a pandas Series is provided, it will be
        converted to a list of strings.
    output_dir : str
    measures : list, optional
        List of measures to get, by default None
    hemis : list, optional
        List of hemispheres to get, by default None
    """
    # Convert pandas Series to list if needed
    if isinstance(subjects, pd.Series):
        subjects = subjects.tolist()
    # Ensure all elements are strings
    subjects = [str(s) for s in subjects]

    stats: dict[str, Path | list[Path]] = {}
    stats["aseg"] = _get_aseg_stats(subjects, "aseg.csv", output_dir=output_dir)
    stats["aparc"] = _get_aparc_stats(subjects, "aparc.csv", output_dir=output_dir, measures=measures, hemis=hemis)
    return stats


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
                        region_data = data[["subject_id", region_col]].copy()
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
                                text=plot_data["subject_id"],
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
                                text=plot_data["subject_id"],
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
                            text=data["subject_id"],
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
                        text=data[data.columns[0]],  # subject_id column
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

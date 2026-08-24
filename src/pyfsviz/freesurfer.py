"""FreeSurfer data."""

from __future__ import annotations

import datetime
import inspect
import logging
import os
import shutil
import textwrap
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import fsqc
import fsqc.createScreenshots
import numpy as np
import pandas as pd
from fsqc import fsqcMain
from importlib_resources import files
from matplotlib import colors
from matplotlib import pyplot as plt
from nilearn import plotting
from nipype.interfaces.freesurfer import MRIConvert
from nipype.interfaces.fsl import FLIRT
from nireports.interfaces.reporting.base import SimpleBeforeAfterRPT

from pyfsviz.reports import Template
from pyfsviz.stats import (
    _comparison_metric_label,
    check_metrics,
    compare_group_metrics,  # noqa: F401
    gen_group_comparison_plots,
    gen_metric_plots,
    get_stats,
    summarize_outlier_subjects,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

_GroupSpec = list[str] | str | Path | None
_GroupDefinition = Mapping[str, _GroupSpec] | list[str]

# Upstream fsqc warns on ambiguous surface contour segments but does not advance
# sortIdx, which hangs createScreenshots. Resolve by taking the first match.
_FSQC_SURFACE_HANG = """\
                    elif findIdx.shape[0] > 1:
                        warnings.warn(
                            "WARNING: a problem occurred with the surface overlays",
                            stacklevel = 2
                        )"""
_FSQC_SURFACE_HANG_FIX = """\
                    elif findIdx.shape[0] > 1:
                        warnings.warn(
                            "WARNING: a problem occurred with the surface overlays",
                            stacklevel = 2
                        )
                        if findIdx[0, 1] == 0:
                            tmpxSort = np.append(
                                tmpxSort,
                                np.array(tmpx[sortIdx[findIdx[0, 0]], ::1], ndmin=2),
                                axis=0,
                            )
                            tmpySort = np.append(
                                tmpySort,
                                np.array(tmpy[sortIdx[findIdx[0, 0]], ::1], ndmin=2),
                                axis=0,
                            )
                        elif findIdx[0, 1] == 1:
                            tmpxSort = np.append(
                                tmpxSort,
                                np.array(tmpx[sortIdx[findIdx[0, 0]], ::-1], ndmin=2),
                                axis=0,
                            )
                            tmpySort = np.append(
                                tmpySort,
                                np.array(tmpy[sortIdx[findIdx[0, 0]], ::-1], ndmin=2),
                                axis=0,
                            )
                        sortIdx = np.delete(sortIdx, findIdx[0, 0])"""


@contextmanager
def _subjects_dir_env(subjects_dir: Path) -> Iterator[None]:
    original = os.environ.get("SUBJECTS_DIR")
    os.environ["SUBJECTS_DIR"] = str(subjects_dir)
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("SUBJECTS_DIR", None)
        else:
            os.environ["SUBJECTS_DIR"] = original


@contextmanager
def _fsqc_screenshots_no_hang() -> Iterator[None]:
    """Patch fsqc screenshot contour sorting so ambiguous segments cannot hang."""
    original = fsqc.createScreenshots.createScreenshots
    source = textwrap.dedent(inspect.getsource(original))
    if _FSQC_SURFACE_HANG not in source:
        logging.getLogger(__name__).warning(
            "Could not patch fsqc surface-overlay hang; screenshots may stall "
            "if contour sorting hits an ambiguous segment",
        )
        yield
        return

    namespace: dict[str, object] = {}
    exec(  # noqa: S102
        source.replace(_FSQC_SURFACE_HANG, _FSQC_SURFACE_HANG_FIX),
        fsqc.createScreenshots.__dict__,
        namespace,
    )
    patched = namespace["createScreenshots"]
    fsqc.createScreenshots.createScreenshots = patched
    fsqcMain.createScreenshots = patched
    try:
        yield
    finally:
        fsqc.createScreenshots.createScreenshots = original
        fsqcMain.createScreenshots = original


def get_freesurfer_colormap(freesurfer_home: Path | str) -> colors.ListedColormap:
    """Generate matplotlib colormap from FreeSurfer LUT.

    Code from:
    https://github.com/Deep-MI/qatools-python/blob/freesurfer-module-releases/qatoolspython/createScreenshots.py

    Parameters
    ----------
    freesurfer_home : path or str representing a path to a directory
        Path corresponding to FREESURFER_HOME env var.

    Returns
    -------
    colormap : matplotlib.colors.ListedColormap
        A matplotlib compatible FreeSurfer colormap.

    """
    freesurfer_home = Path(freesurfer_home) if isinstance(freesurfer_home, str) else freesurfer_home
    # FreeSurfer 8+ appends an optional 7th column (tissue class). Allow up to
    # 8 fields so pandas does not fail on "Expected 6 fields ..., saw 7", then
    # keep only index, name, R, G, B, A for the colormap.
    lut = pd.read_csv(
        freesurfer_home / "FreeSurferColorLUT.txt",
        sep=r"\s+",
        comment="#",
        header=None,
        names=range(8),
        skipinitialspace=True,
        skip_blank_lines=True,
    )
    lut = lut.iloc[:, :6].dropna(subset=[0, 2, 3, 4, 5])
    lut = np.array(lut)
    lut_tab = np.array(lut[:, (2, 3, 4, 5)].astype(float) / 255, dtype="float32")
    lut_tab[:, 3] = 1

    return colors.ListedColormap(lut_tab)


def _html_id(*parts: str) -> str:
    chunks: list[str] = []
    for part in parts:
        slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in part)
        while "--" in slug:
            slug = slug.replace("--", "-")
        slug = slug.strip("-")
        if slug:
            chunks.append(slug)
    return "-".join(chunks)


def _figure_metric_meta(fig: Any) -> dict[str, str]:
    meta = getattr(fig.layout, "meta", None)
    if meta is None:
        return {}
    if not isinstance(meta, dict):
        to_json = getattr(meta, "to_plotly_json", None)
        meta = to_json() if callable(to_json) else dict(meta)
    if not isinstance(meta, dict):
        return {}
    return {
        "metric": str(meta.get("metric") or ""),
        "label": str(meta.get("label") or ""),
    }


def _figure_to_html(fig: Any, *, include_plotlyjs: bool | str) -> str:
    """Render a Plotly figure as an HTML snippet."""
    return fig.to_html(
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        config={"responsive": True},
        default_width="100%",
        default_height=420,
    )


def _plot_sections(
    figures: list[Any],
    *,
    prefix: str,
    include_plotlyjs_first: bool = True,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    include_plotlyjs: bool | str = "cdn" if include_plotlyjs_first else False
    for fig in figures:
        meta = _figure_metric_meta(fig)
        metric = meta.get("metric") or "other"
        label = meta.get("label") or metric.replace("_", " ").title()
        if metric not in grouped:
            grouped[metric] = {
                "id": _html_id(prefix, metric),
                "label": label,
                "plots": [],
            }
        grouped[metric]["plots"].append(
            _figure_to_html(fig, include_plotlyjs=include_plotlyjs),
        )
        include_plotlyjs = False

    sections: list[dict[str, Any]] = []
    for info in grouped.values():
        plots = info["plots"]
        sections.append(
            {
                "id": info["id"],
                "label": info["label"],
                "n_plots": len(plots),
                "plots": plots,
            },
        )
    return sections


def _quality_summary_sections(
    quality_summary: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for metric_name, metric_data in quality_summary.items():
        n_outliers = sum(1 for result in metric_data.values() if result.get("status") == "outliers_detected")
        sections.append(
            {
                "id": _html_id("quality", metric_name),
                "label": _comparison_metric_label(metric_name),
                "n_outliers": n_outliers,
                "n_regions": len(metric_data),
                "regions": metric_data,
                "active": False,
            },
        )
    if sections:
        active_index = next(
            (index for index, section in enumerate(sections) if section["n_outliers"] > 0),
            0,
        )
        sections[active_index]["active"] = True
    return sections


class FreeSurfer:
    """Base class for FreeSurfer data."""

    def __init__(
        self,
        freesurfer_home: str | None = None,
        subjects_dir: str | None = None,
        log_level: str = "INFO",
    ):
        """Initialize the FreeSurfer data.

        Parameters
        ----------
        freesurfer_home : str representing a path to a directory
            Path corresponding to FREESURFER_HOME env var.
        subjects_dir : str representing a path to a directory
            Path corresponding to SUBJECTS_DIR env var.
        log_level : str
            Logging level (e.g., "INFO", "DEBUG", "WARNING").
            Default is "INFO".

        Returns
        -------
        None

        """
        # Set up logger
        logging.basicConfig(level=getattr(logging, log_level.upper()))
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        """Logger for the FreeSurfer class."""

        if freesurfer_home is None:
            self.freesurfer_home = Path(os.environ.get("FREESURFER_HOME") or "")
            """Path to the FreeSurfer home directory."""
        else:
            self.freesurfer_home = Path(freesurfer_home)
            """Path to the FreeSurfer home directory."""
        if not self.freesurfer_home.exists():
            raise FileNotFoundError(
                f"FREESURFER_HOME not found: {self.freesurfer_home}",
            )
        if self.freesurfer_home is None:
            raise ValueError("FREESURFER_HOME must be set")

        if subjects_dir is None:
            self.subjects_dir = Path(os.environ.get("SUBJECTS_DIR") or "")
            """Path to the subjects directory."""
        else:
            self.subjects_dir = Path(subjects_dir)
            """Path to the subjects directory."""
        if not self.subjects_dir.exists():
            raise FileNotFoundError(f"SUBJECTS_DIR not found: {self.subjects_dir}")
        """Path to the subjects directory."""
        self._mni_nii = files("pyfsviz._internal") / "mni305.cor.nii.gz"
        """Path to the MNI template NIfTI file."""
        self._mni_mgz = files("pyfsviz._internal") / "mni305.cor.mgz"
        """Path to the MNI template MGH file."""

    def get_colormap(self) -> colors.ListedColormap:
        """Return the colormap for the FreeSurfer data."""
        return get_freesurfer_colormap(self.freesurfer_home)

    def get_subjects(self, search_dir: str | Path | None = None) -> list[str]:
        """Return FreeSurfer subjects found in a directory.

        Parameters
        ----------
        search_dir
            Directory to scan for subject folders. Defaults to ``self.subjects_dir``.

        Returns
        -------
        list[str]
            Subject IDs with a completed talairach transform.
        """
        base = self.subjects_dir if search_dir is None else Path(search_dir)
        if not base.exists():
            msg = f"Subject search directory not found: {base}"
            raise FileNotFoundError(msg)

        return sorted(
            subject.name
            for subject in base.iterdir()
            if subject.is_dir() and (subject / "mri" / "transforms" / "talairach.lta").exists()
        )

    def resolve_groups(self, groups: _GroupDefinition) -> dict[str, list[str]]:
        """Resolve group definitions to subject ID lists from FreeSurfer directories.

        Each group can be defined as:

        - A list of subject IDs (explicit membership)
        - ``None`` or omitted value: scan ``subjects_dir / group_name``
        - A path string or ``Path``: scan that directory for subject folders

        Parameters
        ----------
        groups
            Group names mapped to subject lists or directories, or a list of
            subdirectory names under ``subjects_dir``.

        Returns
        -------
        dict[str, list[str]]
            Mapping of group names to discovered subject IDs.
        """
        resolved: dict[str, list[str]] = {}
        group_items: list[tuple[str, _GroupSpec]] = (
            [(name, None) for name in groups] if isinstance(groups, list) else list(groups.items())
        )

        for group_name, group_spec in group_items:
            if isinstance(group_spec, list):
                resolved[group_name] = group_spec
                continue

            if group_spec is None or group_spec == "":
                search_dir = self.subjects_dir / group_name
            else:
                search_path = Path(group_spec)
                search_dir = search_path if search_path.is_absolute() else self.subjects_dir / search_path

            subjects = self.get_subjects(search_dir)
            if not subjects:
                self.logger.warning(
                    f"No FreeSurfer subjects found for group '{group_name}' in {search_dir}",
                )
            resolved[group_name] = subjects

        return resolved

    def _group_search_dirs(self, groups: _GroupDefinition) -> dict[str, Path]:
        """Return the FreeSurfer SUBJECTS_DIR to use for each group."""
        search_dirs: dict[str, Path] = {}
        group_items: list[tuple[str, _GroupSpec]] = (
            [(name, None) for name in groups] if isinstance(groups, list) else list(groups.items())
        )

        for group_name, group_spec in group_items:
            if isinstance(group_spec, list):
                search_dirs[group_name] = self.subjects_dir
            elif group_spec is None or group_spec == "":
                search_dirs[group_name] = self.subjects_dir / group_name
            else:
                search_path = Path(group_spec)
                search_dirs[group_name] = search_path if search_path.is_absolute() else self.subjects_dir / search_path

        return search_dirs

    def _stats_output_files(
        self,
        stats: dict[str, Path | list[Path]],
        *,
        prefix: str = "",
    ) -> list[Path]:
        stats_files: list[Path] = []
        name_suffix = f"_{prefix}" if prefix else ""

        aseg_value = stats.get("aseg")
        if isinstance(aseg_value, Path):
            if name_suffix:
                dest = aseg_value.with_name(
                    f"{aseg_value.stem}{name_suffix}{aseg_value.suffix}",
                )
                shutil.copy2(aseg_value, dest)
                stats_files.append(dest)
            else:
                stats_files.append(aseg_value)

        aparc_value = stats.get("aparc")
        if isinstance(aparc_value, list):
            for aparc_file in aparc_value:
                if name_suffix and "combined" not in aparc_file.stem:
                    dest = aparc_file.with_name(
                        f"{aparc_file.stem}{name_suffix}{aparc_file.suffix}",
                    )
                    shutil.copy2(aparc_file, dest)
                    stats_files.append(dest)
                elif "combined" not in aparc_file.stem:
                    stats_files.append(aparc_file)
        elif isinstance(aparc_value, Path):
            stats_files.append(aparc_value)

        return stats_files

    def _collect_group_stats_files(
        self,
        output_dir: Path,
        subjects: list[str],
        groups: dict[str, list[str]] | None,
        group_search_dirs: dict[str, Path] | None,
    ) -> list[Path]:
        if groups is None or group_search_dirs is None:
            stats = get_stats(subjects, str(output_dir))
            return self._stats_output_files(stats)

        unique_dirs = {search_dir.resolve() for search_dir in group_search_dirs.values()}
        if len(unique_dirs) == 1 and next(iter(unique_dirs)) == self.subjects_dir.resolve():
            stats = get_stats(subjects, str(output_dir))
            return self._stats_output_files(stats)

        stats_files: list[Path] = []
        for group_name, group_subjects in groups.items():
            if not group_subjects:
                continue
            group_dir = group_search_dirs[group_name]
            group_out = output_dir / f"group_{group_name}"
            group_out.mkdir(parents=True, exist_ok=True)
            with _subjects_dir_env(group_dir):
                stats = get_stats(group_subjects, str(group_out))
            stats_files.extend(self._stats_output_files(stats, prefix=group_name))

        return stats_files

    def check_recon_all(self, subject: str) -> bool:
        """Verify that the subject's FreeSurfer recon finished successfully."""
        recon_file = self.subjects_dir / subject / "scripts" / "recon-all.log"

        with open(recon_file, encoding="utf-8") as f:
            line = f.readlines()[-1]
            return "finished without error" in line

    def gen_tlrc_data(self, subject: str, output_dir: str) -> None:
        """Generate inverse talairach data for report generation.

        Parameters
        ----------
        output_dir : str
            Path for intermediate file output.

        Examples
        --------
        >>> from pyfsviz.freesurfer import FreeSurfer
        >>> fs_dir = FreeSurfer(
        ...     freesurfer_home="/opt/freesurfer",
        ...     subjects_dir="/opt/data",
        ... )
        >>> fs_dir.gen_tlrc_data("sub-001", "/opt/data/sub-001/mri/transforms")
        """
        # get inverse transform
        lta_file = self.subjects_dir / subject / "mri" / "transforms" / "talairach.xfm.lta"
        xfm = np.genfromtxt(lta_file, skip_header=5, max_rows=4)
        inverse_xfm = np.linalg.inv(xfm)
        np.savetxt(
            f"{output_dir}/inv.xfm",
            inverse_xfm,
            fmt="%0.8f",
            delimiter=" ",
            newline="\n",
            encoding="utf-8",
        )

        # convert subject original T1 to nifti (for FSL)
        convert = MRIConvert(
            in_file=self.subjects_dir / subject / "mri" / "orig.mgz",
            out_file=f"{output_dir}/orig.nii.gz",
            out_type="niigz",
        )
        convert.run()

        # use FSL to convert template file to subject original space
        flirt = FLIRT(
            in_file=self._mni_nii,
            reference=f"{output_dir}/orig.nii.gz",
            out_file=f"{output_dir}/mni2orig.nii.gz",
            in_matrix_file=f"{output_dir}/inv.xfm",
            apply_xfm=True,
            out_matrix_file=f"{output_dir}/out.mat",
        )
        flirt.run()

    def gen_tlrc_report(
        self,
        subject: str,
        output_dir: str,
        tlrc_dir: str | None = None,
        *,
        gen_data: bool = True,
    ) -> Path:
        """Generate a before and after report of Talairach registration. (Will also run file generation if needed).

        Parameters
        ----------
        subject : str
            Subject ID.
        output_dir : str
            Path to SVG output.
        gen_data : bool
            Generate inverse Talairach data, by default True
        tlrc_dir : str | None
            Path to output of `gen_tlrc_data`. Default is the subject's mri/transforms directory.

        Returns
        -------
        Path:
            SVG file generated from the niworkflows SimpleBeforeAfterRPT

        Examples
        --------
        >>> from pyfsviz.freesurfer import FreeSurfer
        >>> fs_dir = FreeSurfer(
        ...     freesurfer_home="/opt/freesurfer",
        ...     subjects_dir="/opt/data",
        ... )
        >>> report = fs_dir.gen_tlrc_report("sub-001", "/opt/data/reports/sub-001")
        """
        if tlrc_dir is None:
            tlrc_dir = f"{self.subjects_dir}/{subject}/mri/transforms"

        mri_dir = f"{self.subjects_dir}/{subject}/mri"

        if gen_data:
            self.gen_tlrc_data(subject, tlrc_dir)

        # use white matter segmentation to compare registrations
        report = SimpleBeforeAfterRPT(
            before=f"{mri_dir}/orig.mgz",
            after=f"{tlrc_dir}/mni2orig.nii.gz",
            wm_seg=f"{mri_dir}/wm.mgz",
            before_label="Subject Orig",
            after_label="Template",
            out_report=f"{output_dir}/tlrc.svg",
        )
        result = report.run()
        return result.outputs.out_report

    def gen_aparcaseg_plots(self, subject: str, output_dir: str) -> Path:
        """Generate parcellation images (aparc & aseg) and return the path to the aparcaseg.png file.

        Parameters
        ----------
        subject : str
            Subject ID.
        output_dir : str
            Path to output directory.

        Returns
        -------
        Path:
            Path to the aparcaseg.png file.

        Examples
        --------
        >>> from pyfsviz.freesurfer import FreeSurfer
        >>> fs_dir = FreeSurfer(
        ...     freesurfer_home="/opt/freesurfer",
        ...     subjects_dir="/opt/data",
        ... )
        >>> images = fs_dir.gen_aparcaseg_plots("sub-001", "/opt/data/reports/sub-001")
        """
        with _fsqc_screenshots_no_hang():
            fsqc.run_fsqc(
                subjects_dir=str(self.subjects_dir),
                output_dir=output_dir,
                subjects=[subject],
                screenshots=True,
                screenshots_overlay="aparc+aseg.mgz",
                screenshots_views=[
                    "x=-40",
                    "x=-30",
                    "x=-20",
                    "x=-10",
                    "x=0",
                    "x=10",
                    "x=20",
                    "x=30",
                    "x=40",
                    "y=-40",
                    "y=-30",
                    "y=-20",
                    "y=-10",
                    "y=0",
                    "y=10",
                    "y=20",
                    "y=30",
                    "y=40",
                    "z=-40",
                    "z=-30",
                    "z=-20",
                    "z=-10",
                    "z=0",
                    "z=10",
                    "z=20",
                    "z=30",
                    "z=40",
                ],
                screenshots_layout=["3", "9"],
                no_group=True,
            )

        # Clean up/move files
        shutil.move(
            f"{output_dir}/screenshots/{subject}/{subject}.png",
            f"{output_dir}/aparcaseg.png",
        )
        shutil.move(
            f"{output_dir}/metrics/{subject}/metrics.csv",
            f"{output_dir}/metrics.csv",
        )
        shutil.rmtree(f"{output_dir}/screenshots")
        shutil.rmtree(f"{output_dir}/status")
        shutil.rmtree(f"{output_dir}/metrics")

        return Path(f"{output_dir}/aparcaseg.png")

    def gen_surf_plots(self, subject: str, output_dir: str) -> list[Path]:
        """Generate pial, inflated, and sulcal images from various viewpoints.

        Parameters
        ----------
        output_dir : str
            Surface plot output directory.

        Returns
        -------
        list[Path]:
            List of generated PNG images

        Examples
        --------
        >>> from pyfsviz.freesurfer import FreeSurfer
        >>> fs_dir = FreeSurfer(
        ...     freesurfer_home="/opt/freesurfer",
        ...     subjects_dir="/opt/data",
        ... )
        >>> images = fs_dir.gen_surf_plots("sub-001", "/opt/data/reports/sub-001")
        """
        surf_dir = f"{self.subjects_dir}/{subject}/surf"
        label_dir = f"{self.subjects_dir}/{subject}/label"
        cmap = self.get_colormap()
        generated: list[Path] = []

        hemis = {"lh": "left", "rh": "right"}
        for key, val in hemis.items():
            pial = f"{surf_dir}/{key}.pial"
            inflated = f"{surf_dir}/{key}.inflated"
            sulc = f"{surf_dir}/{key}.sulc"
            white = f"{surf_dir}/{key}.white"
            annot = f"{label_dir}/{key}.aparc.annot"

            label_files = {pial: "pial", inflated: "infl", white: "white"}

            for surf, label in label_files.items():
                fig, axs = plt.subplots(2, 3, subplot_kw={"projection": "3d"})
                plotting.plot_surf_roi(
                    surf,
                    annot,
                    hemi=val,
                    view="lateral",
                    bg_map=sulc,
                    bg_on_data=True,
                    darkness=1,
                    cmap=cmap,
                    axes=axs[0, 0],
                    figure=fig,
                    colorbar=False,
                )
                plotting.plot_surf_roi(
                    surf,
                    annot,
                    hemi=val,
                    view="medial",
                    bg_map=sulc,
                    bg_on_data=True,
                    darkness=1,
                    cmap=cmap,
                    axes=axs[0, 1],
                    figure=fig,
                    colorbar=False,
                )
                plotting.plot_surf_roi(
                    surf,
                    annot,
                    hemi=val,
                    view="dorsal",
                    bg_map=sulc,
                    bg_on_data=True,
                    darkness=1,
                    cmap=cmap,
                    axes=axs[0, 2],
                    figure=fig,
                    colorbar=False,
                )
                plotting.plot_surf_roi(
                    surf,
                    annot,
                    hemi=val,
                    view="ventral",
                    bg_map=sulc,
                    bg_on_data=True,
                    darkness=1,
                    cmap=cmap,
                    axes=axs[1, 0],
                    figure=fig,
                    colorbar=False,
                )
                plotting.plot_surf_roi(
                    surf,
                    annot,
                    hemi=val,
                    view="anterior",
                    bg_map=sulc,
                    bg_on_data=True,
                    darkness=1,
                    cmap=cmap,
                    axes=axs[1, 1],
                    figure=fig,
                    colorbar=False,
                )
                plotting.plot_surf_roi(
                    surf,
                    annot,
                    hemi=val,
                    view="posterior",
                    bg_map=sulc,
                    bg_on_data=True,
                    darkness=1,
                    cmap=cmap,
                    axes=axs[1, 2],
                    figure=fig,
                    colorbar=False,
                )

                out_file = Path(output_dir) / f"{key}_{label}.png"
                plt.savefig(out_file, dpi=300, format="png")
                plt.close()
                generated.append(out_file)

        return sorted(generated)

    def gen_html_report(
        self,
        subject: str,
        output_dir: str,
        img_list: list[Path] | None = None,
        template: str | None = None,
    ) -> Path:
        """Generate html report with FreeSurfer images.

        Parameters
        ----------
        subject : str
            Subject ID.
        output_dir : str
            HTML file name
        img_list : list[Path] | None
            List of image paths (PNG format).
        template : str | None
            HTML template to use. Default is local freesurfer.html.

        Returns
        -------
        Path:
            Path to html file.

        Examples
        --------
        >>> from pyfsviz.freesurfer import FreeSurfer
        >>> fs_dir = FreeSurfer(
        ...     freesurfer_home="/opt/freesurfer",
        ...     subjects_dir="/opt/data",
        ... )
        >>> report = fs_dir.gen_html_report("sub-001", "/opt/data/reports")
        """
        if template is None:
            template = str(files("pyfsviz._internal.html") / "individual.html")
        if img_list is None:
            img_list = list((self.subjects_dir / subject).glob("**/*.{png,svg}"))

        tlrc = []
        aseg = []
        surf = []

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Subject-specific directory
        subject_dir = output_path / subject
        subject_dir.mkdir(parents=True, exist_ok=True)

        for img in img_list:
            if "tlrc" in img.name and img.suffix == ".svg":
                # Read SVG content directly for embedding
                with open(img, encoding="utf-8") as f:
                    svg_content = f.read()
                tlrc.append(svg_content)
            # Images are already in the subject directory, just reference by filename
            elif "aparcaseg" in img.name:
                aseg.append(img.name)
            else:
                labels = {
                    "lh_pial": "LH Pial",
                    "rh_pial": "RH Pial",
                    "lh_infl": "LH Inflated",
                    "rh_infl": "RH Inflated",
                    "lh_white": "LH White Matter",
                    "rh_white": "RH White Matter",
                }
                surface_type = img.stem
                surf_tuple = (labels.get(surface_type, surface_type), img.name)
                surf.append(surf_tuple)

        # Read metrics.csv if it exists
        metrics = None
        metrics_csv_path = output_path / "metrics.csv"
        if metrics_csv_path.exists():
            try:
                df = pd.read_csv(metrics_csv_path)
                # Filter for current subject if subject column exists
                if "subject" in df.columns:
                    subject_data = df[df["subject"] == subject]
                    if not subject_data.empty:
                        metrics = subject_data.iloc[0].to_dict()
                # If no subject column, assume single row
                elif len(df) > 0:
                    metrics = df.iloc[0].to_dict()
                # Replace NaN values with None for proper Jinja2 handling
                if metrics:
                    metrics = {k: (None if pd.isna(v) else v) for k, v in metrics.items()}
            except (
                pd.errors.EmptyDataError,
                pd.errors.ParserError,
                UnicodeDecodeError,
                PermissionError,
                OSError,
            ) as e:
                self.logger.warning(f"Could not read metrics.csv: {e}")

        _config = {
            "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).strftime(
                "%Y-%m-%d, %H:%M",
            ),
            "subject": subject,
            "tlrc": tlrc,
            "aseg": aseg,
            "surf": surf,
            "metrics": metrics,
        }

        # Save HTML file in subject directory
        html_file = subject_dir / f"{subject}.html"
        tpl = Template(str(template))
        tpl.generate_conf(_config, str(html_file))

        return html_file

    def gen_batch_reports(
        self,
        output_dir: str | Path,
        subjects: list[str] | None = None,
        template: str | None = None,
        *,
        gen_images: bool = True,
        skip_failed: bool = True,
        skip_existing: bool = False,
    ) -> dict[str, Path | Exception]:
        """Generate HTML reports with images for multiple subjects.

        This method first generates all required images (TLRC, aparc+aseg, surfaces)
        and then creates HTML reports for each subject.

        Parameters
        ----------
        output_dir : str or Path
            Directory where HTML reports will be saved.
        subjects : list[str] or None
            List of subject IDs to process. If None, processes all subjects
            in the subjects directory.
        template : str or None
            HTML template to use. Default is local individual.html.
        gen_images : bool
            Generate images for each subject. Default is True.
        skip_failed : bool
            If True, continues processing other subjects if one fails.
            If False, raises exception on first failure.
        skip_existing : bool
            If True, skip subjects that already have an HTML report at
            ``{output_dir}/{subject}/{subject}.html``. Incomplete subjects
            (images but no report) are still processed. Default is False.

        Returns
        -------
        dict[str, Path | Exception]
            Dictionary mapping subject IDs to either the generated (or existing)
            HTML file path or the exception that occurred during processing.

        Examples
        --------
        >>> from pyfsviz.freesurfer import FreeSurfer
        >>> fs = FreeSurfer()
        >>> results = fs.gen_batch_reports("reports/")
        >>> for subject, result in results.items():
        ...     if isinstance(result, Path):
        ...         print(f"Generated report for {subject}: {result}")
        ...     else:
        ...         print(f"Failed to generate report for {subject}: {result}")
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if subjects is None:
            subjects = self.get_subjects()

        self.logger.info(
            f"Generating reports with images for {len(subjects)} subjects...",
        )
        self.logger.info(f"Output directory: {output_dir}")

        results: dict[str, Path | Exception] = {}
        skipped = 0

        for i, subject in enumerate(subjects, 1):
            existing_html = output_dir / subject / f"{subject}.html"
            if skip_existing and existing_html.is_file():
                self.logger.info(
                    f"[{i}/{len(subjects)}] Skipping {subject}: report already exists",
                )
                results[subject] = existing_html
                skipped += 1
                continue

            self.logger.info(f"[{i}/{len(subjects)}] Processing subject: {subject}")

            try:
                # Check if recon-all completed successfully
                if not self.check_recon_all(subject):
                    self.logger.warning(
                        f"Subject {subject} recon-all did not complete successfully",
                    )

                # Create subject-specific output directory for images
                subject_output_dir = output_dir / subject
                subject_output_dir.mkdir(parents=True, exist_ok=True)

                # Generate images
                self.logger.info(f"  Generating images for {subject}...")

                img_list = []
                if gen_images:
                    # Generate TLRC data and report
                    # Use a temporary subdirectory for intermediate files
                    temp_tlrc_dir = subject_output_dir / "tlrc_temp"
                    temp_tlrc_dir.mkdir(exist_ok=True)

                    self.gen_tlrc_data(subject, str(temp_tlrc_dir))
                    tlrc = Path(self.gen_tlrc_report(subject, str(temp_tlrc_dir)))

                    # Move tlrc.svg to subject directory
                    if tlrc.exists():
                        new_tlrc_path = subject_output_dir / "tlrc.svg"
                        tlrc.rename(new_tlrc_path)
                        img_list.append(new_tlrc_path)
                    else:
                        img_list.append(tlrc)

                    # Clean up intermediate files
                    shutil.rmtree(temp_tlrc_dir, ignore_errors=True)

                    # Generate aparc+aseg plots - save directly to subject directory
                    aparcaseg = self.gen_aparcaseg_plots(
                        subject,
                        str(subject_output_dir),
                    )
                    img_list.append(aparcaseg)

                    # Generate surface plots - save directly to subject directory
                    surf = self.gen_surf_plots(subject, str(subject_output_dir))
                    img_list.extend(surf)
                else:
                    img_list = list(subject_output_dir.glob("**/*.{png,svg}"))

                # Generate HTML report using all generated images
                html_file = self.gen_html_report(
                    subject=subject,
                    output_dir=str(output_dir),
                    img_list=img_list,
                    template=template,
                )

                results[subject] = html_file

                self.logger.info(f"  ✓ Generated report with images: {html_file}")

            except Exception as e:
                error_msg = f"Failed to generate report with images for {subject}: {e!s}"
                results[subject] = e

                self.logger.error(f"  ✗ {error_msg}")  # noqa: TRY400

                if not skip_failed:
                    raise e  # noqa: TRY201 # pylint: disable=try-except-raise

        successful = sum(1 for result in results.values() if isinstance(result, Path)) - skipped
        failed = len(results) - successful - skipped
        self.logger.info("\nBatch report generation with images completed:")
        self.logger.info(f"  Successful: {successful}")
        self.logger.info(f"  Skipped: {skipped}")
        self.logger.info(f"  Failed: {failed}")

        return results

    def gen_group_report(
        self,
        output_dir: str | Path,
        subjects: list[str] | None = None,
        groups: _GroupDefinition | None = None,
        template: str | None = None,
        *,
        sd_threshold: float = 3.0,
    ) -> Path:
        """Generate a group report with outlier information for multiple subjects.

        Parameters
        ----------
        output_dir : str or Path
            Directory where HTML report will be saved.
        subjects : list[str] | None
            List of subject IDs to process. If None, processes all subjects
            in the subjects directory.
        groups : _GroupDefinition | None
            Optional group definitions for between-group box plots. Each group
            can be a list of subject IDs or a FreeSurfer directory to scan:

            - ``["control", "patient"]`` scans ``subjects_dir/control`` and
              ``subjects_dir/patient``
            - ``{"control": None, "patient": None}`` same as above
            - ``{"control": "/path/to/control_cohort"}`` scans an explicit path
            - ``{"control": ["sub-001"]}`` uses explicit subject IDs
        template : str | None
            HTML template to use. Default is local group.html.
        sd_threshold : float
            Standard deviation threshold for outlier detection. Default is 3.0.

        Returns
        -------
        Path:
            Path to html file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        group_search_dirs: dict[str, Path] | None = None
        resolved_groups: dict[str, list[str]] | None = None
        if groups is not None:
            if subjects is not None:
                self.logger.warning(
                    "Both subjects and groups were provided; using subjects from groups",
                )
            group_search_dirs = self._group_search_dirs(groups)
            resolved_groups = self.resolve_groups(groups)
            subjects = [subject for group_subjects in resolved_groups.values() for subject in group_subjects]
        elif subjects is None:
            subjects = self.get_subjects()

        self.logger.info(f"Generating group report for {len(subjects)} subjects...")
        if resolved_groups:
            for group_name, group_subjects in resolved_groups.items():
                self.logger.info(
                    f"  Group {group_name}: {len(group_subjects)} subject(s)",
                )
        self.logger.info(f"Output directory: {output_dir}")

        stats_files = self._collect_group_stats_files(
            output_dir,
            subjects,
            resolved_groups,
            group_search_dirs,
        )

        # Generate plots
        metric_figures = gen_metric_plots(stats_files)
        comparison_figures = gen_group_comparison_plots(stats_files, resolved_groups) if resolved_groups else []
        comparison_plots = _plot_sections(
            comparison_figures,
            prefix="cmp",
            include_plotlyjs_first=bool(comparison_figures),
        )
        plot_sections = _plot_sections(
            metric_figures,
            prefix="plots",
            include_plotlyjs_first=not comparison_figures,
        )
        plot_htmls = [html for section in plot_sections for html in section["plots"]]

        quality_summary = check_metrics(stats_files, sd_threshold=sd_threshold)
        quality_sections = _quality_summary_sections(quality_summary)
        outlier_subjects = summarize_outlier_subjects(quality_summary)

        # Prepare template config
        if template is None:
            template = str(files("pyfsviz._internal.html") / "group.html")

        _config = {
            "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).strftime(
                "%Y-%m-%d, %H:%M",
            ),
            "subjects": subjects,
            "num_subjects": len(subjects),
            "groups": resolved_groups,
            "quality_summary": quality_summary,
            "quality_sections": quality_sections,
            "outlier_subjects": outlier_subjects,
            "plots": plot_htmls,
            "plot_sections": plot_sections,
            "comparison_plots": comparison_plots,
            "sd_threshold": sd_threshold,
        }

        # Generate HTML file
        html_file = output_dir / "group_report.html"
        tpl = Template(str(template))
        tpl.generate_conf(_config, str(html_file))

        self.logger.info(f"✓ Generated group report: {html_file}")
        return html_file

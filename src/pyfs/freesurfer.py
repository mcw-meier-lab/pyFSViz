"""FreeSurfer data."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colors


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
    lut = pd.read_csv(
        freesurfer_home / "FreeSurferColorLUT.txt",
        sep=r"\s+",
        comment="#",
        header=None,
        skipinitialspace=True,
        skip_blank_lines=True,
    )
    lut = np.array(lut)
    lut_tab = np.array(lut[:, (2, 3, 4, 5)] / 255, dtype="float32")
    lut_tab[:, 3] = 1

    return colors.ListedColormap(lut_tab)


class FreeSurfer:
    """Base class for FreeSurfer data."""

    def __init__(
        self,
        freesurfer_home: str | None = None,
        subjects_dir: str | None = None,
    ):
        """Initialize the FreeSurfer data.

        Parameters
        ----------
        freesurfer_home : path or str representing a path to a directory
        Path corresponding to FREESURFER_HOME env var.
        subjects_dir : path or str representing a path to a directory
        Path corresponding to SUBJECTS_DIR env var.

        Returns
        -------
        None

        """
        if freesurfer_home is None:
            self.freesurfer_home = Path(os.environ.get("FREESURFER_HOME") or "")
        else:
            self.freesurfer_home = Path(freesurfer_home)
        if not self.freesurfer_home.exists():
            raise FileNotFoundError(f"FREESURFER_HOME not found: {self.freesurfer_home}")
        if self.freesurfer_home is None:
            raise ValueError("FREESURFER_HOME must be set")

        if subjects_dir is None:
            self.subjects_dir = Path(os.environ.get("SUBJECTS_DIR") or "")
        else:
            self.subjects_dir = Path(subjects_dir)
        if not self.subjects_dir.exists():
            raise FileNotFoundError(f"SUBJECTS_DIR not found: {self.subjects_dir}")

    def get_colormap(self) -> colors.ListedColormap:
        """Return the colormap for the FreeSurfer data."""
        return get_freesurfer_colormap(self.freesurfer_home)

    def get_subjects(self) -> list[str]:
        """Return the subjects in the subjects directory."""
        return [
            subject.name
            for subject in self.subjects_dir.iterdir()
            if subject.is_dir() and (subject / "mri" / "transforms" / "talairach.lta").exists()
        ]

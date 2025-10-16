"""FreeSurfer data."""

from __future__ import annotations

import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
from importlib_resources import files
from matplotlib import colors
from matplotlib import pyplot as plt
from nilearn import plotting
from nipype.interfaces.freesurfer import MRIConvert
from nipype.interfaces.fsl import FLIRT
from nireports.interfaces.reporting.base import SimpleBeforeAfterRPT

from pyfs.reports import Template


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

        self.mni_nii = files("pyfs._internal") / "mni305.cor.nii.gz"
        self.mni_mgz = files("pyfs._internal") / "mni305.cor.mgz"

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
        >>> from pyfs.freesurfer import FreeSurfer
        >>> fs_dir = FreeSurfer(
        ...     freesurfer_home="/opt/freesurfer",
        ...     subjects_dir="/opt/data",
        ...     subject="sub-001",
        ... )
        >>> fs_dir.gen_tlrc_data("sub-001", Path("/opt/data/sub-001/mri/transforms"))
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
            in_file=self.mni_nii,
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
        >>> from pyfs.freesurfer import FreeSurfer
        >>> fs_dir = FreeSurfer(
        ...     freesurfer_home="/opt/freesurfer",
        ...     subjects_dir="/opt/data",
        ...     subject="sub-001",
        ... )
        >>> report = fs_dir.gen_tlrc_report(
        ...     "sub-001", Path("/opt/data/sub-001/mri/transforms")
        ... )
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

    def gen_aparcaseg_plots(self, subject: str, output_dir: str, num_imgs: int = 10) -> list[Path]:
        """Generate parcellation images (aparc & aseg).

        Parameters
        ----------
        output_dir : str
            Path to output directory.
        num_imgs : int
            Number of images/slices to make.

        Returns
        -------
        list
            List of SVG image paths

        Examples
        --------
        >>> from pyfs.freesurfer import FreeSurfer
        >>> fs_dir = FreeSurfer(
        ...     freesurfer_home="/opt/freesurfer",
        ...     subjects_dir="/opt/data",
        ...     subject="sub-001",
        ... )
        >>> images = fs_dir.gen_aparcaseg_plots(
        ...     "sub-001", Path("/opt/data/sub-001/mri/transforms")
        ... )
        """
        mri_dir = f"{self.subjects_dir}/{subject}/mri"
        cmap = self.get_colormap()

        # get parcellation and segmentation images
        plotting.plot_roi(
            f"{mri_dir}/aparc+aseg.mgz",
            f"{mri_dir}/T1.mgz",
            cmap=cmap,
            display_mode="mosaic",
            dim=-1,
            cut_coords=num_imgs,
            alpha=0.5,
            output_file=f"{output_dir}/aseg.svg",
        )
        display = plotting.plot_anat(
            f"{mri_dir}/brainmask.mgz",
            display_mode="mosaic",
            cut_coords=num_imgs,
            dim=-1,
        )
        display.add_contours(
            f"{mri_dir}/lh.ribbon.mgz",
            colors="b",
            linewidths=0.5,
            levels=[0.5],
        )
        display.add_contours(
            f"{mri_dir}/rh.ribbon.mgz",
            colors="r",
            linewidths=0.5,
            levels=[0.5],
        )
        display.savefig(f"{output_dir}/aparc.svg")
        display.close()

        return [Path(f"{output_dir}/aseg.svg"), Path(f"{output_dir}/aparc.svg")]

    def gen_surf_plots(self, subject: str, output_dir: str) -> list[Path]:
        """Generate pial, inflated, and sulcal images from various viewpoints.

        Parameters
        ----------
        output_dir : str
            Surface plot output directory.

        Returns
        -------
        list[Path]:
            List of generated SVG images

        Examples
        --------
        >>> from pyfs.freesurfer import FreeSurfer
        >>> fs_dir = FreeSurfer(
        ...     freesurfer_home="/opt/freesurfer",
        ...     subjects_dir="/opt/data",
        ...     subject="sub-001",
        ... )
        >>> images = fs_dir.gen_surf_plots("sub-001", Path("/opt/data/sub-001/surf"))
        """
        surf_dir = f"{self.subjects_dir}/{subject}/surf"
        label_dir = f"{self.subjects_dir}/{subject}/label"
        cmap = self.get_colormap()

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
                )

                plt.savefig(f"{output_dir}/{key}_{label}.svg", dpi=300, format="svg")
                plt.close()

        return sorted(Path(output_dir).glob("*svg"))

    def gen_html_report(
        self,
        subject: str,
        output_dir: str,
        img_out: str | None = None,
        template: str | None = None,
    ) -> Path:
        """Generate html report with FreeSurfer images.

        Parameters
        ----------
        subject : str
            Subject ID.
        output_dir : str
            HTML file name
        img_out : str | None
            Location where SVG images are saved, default is subject's data directory.
        template : str | None
            HTML template to use. Default is local freesurfer.html.

        Returns
        -------
        Path:
            Path to html file.

        Examples
        --------
        >>> from pyfs.freesurfer import FreeSurfer
        >>> fs_dir = FreeSurfer(
        ...     freesurfer_home="/opt/freesurfer",
        ...     subjects_dir="/opt/data",
        ...     subject="sub-001",
        ... )
        >>> report = fs_dir.gen_html_report(out_name="sub-001.html", output_dir=".")
        """
        if template is None:
            template = files("pyfs._internal.html") / "individual.html"
        if img_out is None:
            image_list = list((self.subjects_dir / subject).glob("*/*svg"))
        else:
            image_list = list(Path(img_out).glob("*svg"))

        tlrc = []
        aseg = []
        surf = []

        for img in image_list:
            with open(img, "r", encoding="utf-8") as img_file:
                img_data = img_file.read()

            if "tlrc" in img.name:
                tlrc.append(img_data)
            elif "aseg" in img.name or "aparc" in img.name:
                aseg.append(img_data)
            else:
                labels = {
                    "lh_pial": "LH Pial",
                    "rh_pial": "RH Pial",
                    "lh_infl": "LH Inflated",
                    "rh_infl": "RH Inflated",
                    "lh_white": "LH White Matter",
                    "rh_white": "RH White Matter",
                }
                surf_tuple = (labels[img.stem], img_data)
                surf.append(surf_tuple)

        _config = {
            "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d, %H:%M"),
            "subject": subject,
            "tlrc": tlrc,
            "aseg": aseg,
            "surf": surf,
        }

        tpl = Template(str(template))
        tpl.generate_conf(_config, f"{output_dir}/{subject}.html")

        return Path(output_dir) / f"{subject}.html"

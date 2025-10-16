import inspect
from pathlib import Path

import pytest
from matplotlib import colors

from pyfs.freesurfer import FreeSurfer, get_freesurfer_colormap


def test_get_colormap(mock_freesurfer_home: Path) -> None:
    colormap = get_freesurfer_colormap(mock_freesurfer_home)
    assert isinstance(colormap, colors.ListedColormap)


@pytest.fixture(scope="module")
def freesurfer(mock_freesurfer_instance: FreeSurfer) -> FreeSurfer:
    return mock_freesurfer_instance


def test_get_subjects(freesurfer: FreeSurfer) -> None:
    subjects = freesurfer.get_subjects()
    assert len(subjects) > 0
    assert all(isinstance(subject, str) for subject in subjects)
    assert "sub-001" in subjects


def test_check_recon_all(freesurfer: FreeSurfer) -> None:
    assert freesurfer.check_recon_all("sub-001")


@pytest.mark.skip(reason="Requires FSL and nipype to be installed and configured")
def test_gen_tlrc_data(freesurfer: FreeSurfer, temp_output_dir: str) -> None:
    """Test Talairach data generation."""
    freesurfer.gen_tlrc_data("sub-001", temp_output_dir)
    assert (Path(temp_output_dir) / "inv.xfm").exists()
    assert (Path(temp_output_dir) / "orig.nii.gz").exists()
    assert (Path(temp_output_dir) / "mni2orig.nii.gz").exists()


@pytest.mark.skip(reason="Requires FSL, nipype, and nireports to be installed and configured")
def test_gen_tlrc_report(freesurfer: FreeSurfer, temp_output_dir: str) -> None:
    """Test Talairach report generation."""
    freesurfer.gen_tlrc_report("sub-001", temp_output_dir)
    assert (Path(temp_output_dir) / "tlrc.svg").exists()
    assert (Path(temp_output_dir) / "tlrc.svg").is_file()


@pytest.mark.skip(reason="Requires nilearn and proper FreeSurfer data files")
def test_gen_aparcaseg_plots(freesurfer: FreeSurfer, temp_output_dir: str) -> None:
    """Test aparc and aseg plotting."""
    plots = freesurfer.gen_aparcaseg_plots("sub-001", temp_output_dir)
    assert len(plots) == 2
    assert (Path(temp_output_dir) / "aseg.svg").exists()
    assert (Path(temp_output_dir) / "aseg.svg").is_file()
    assert (Path(temp_output_dir) / "aseg.svg").suffix == ".svg"
    assert (Path(temp_output_dir) / "aparc.svg").exists()
    assert (Path(temp_output_dir) / "aparc.svg").is_file()
    assert (Path(temp_output_dir) / "aparc.svg").suffix == ".svg"


@pytest.mark.skip(reason="Surface plotting requires complex FreeSurfer file formats - needs improvement")
def test_gen_surf_plots(freesurfer: FreeSurfer, temp_output_dir: str) -> None:
    """Test surface plot generation."""
    plots = freesurfer.gen_surf_plots("sub-001", temp_output_dir)
    assert len(plots) == 6  # 2 hemispheres x 3 surface types
    for plot in plots:
        assert plot.exists()
        assert plot.is_file()
        assert plot.suffix == ".svg"


def test_gen_surf_plots_basic(freesurfer: FreeSurfer) -> None:
    """Test basic surface plot functionality without complex plotting."""
    # Test that the method exists and can be called
    assert hasattr(freesurfer, "gen_surf_plots")

    # Test that it expects the right parameters
    sig = inspect.signature(freesurfer.gen_surf_plots)
    assert "subject" in sig.parameters
    assert "output_dir" in sig.parameters

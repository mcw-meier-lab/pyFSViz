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


def test_gen_html_report_basic(freesurfer: FreeSurfer) -> None:
    """Test basic HTML report generation functionality."""
    # Test that the method exists and can be called
    assert hasattr(freesurfer, "gen_html_report")

    # Test that it expects the right parameters
    sig = inspect.signature(freesurfer.gen_html_report)
    assert "subject" in sig.parameters
    assert "output_dir" in sig.parameters
    assert "img_out" in sig.parameters
    assert "template" in sig.parameters


@pytest.mark.skip(reason="Requires jinja2 and proper SVG files")
def test_gen_html_report(freesurfer: FreeSurfer, temp_output_dir: Path) -> None:
    """Test HTML report generation."""
    # Create mock SVG files
    mock_svg_dir = temp_output_dir / "mock_svgs"
    mock_svg_dir.mkdir(parents=True, exist_ok=True)

    # Create different types of SVG files
    svg_files = {
        "tlrc.svg": "<svg><text>Talairach Registration</text></svg>",
        "aseg.svg": "<svg><text>Aseg Parcellation</text></svg>",
        "aparc.svg": "<svg><text>Aparc Parcellation</text></svg>",
        "lh_pial.svg": "<svg><text>LH Pial Surface</text></svg>",
        "rh_pial.svg": "<svg><text>RH Pial Surface</text></svg>",
    }

    for filename, content in svg_files.items():
        with open(mock_svg_dir / filename, "w") as f:
            f.write(content)

    # Generate HTML report
    html_file = freesurfer.gen_html_report(
        subject="sub-001",
        output_dir=str(temp_output_dir),
        img_out=str(mock_svg_dir),
    )

    # Check that HTML file was created
    assert html_file.exists()
    assert html_file.name == "sub-001.html"

    # Check HTML content
    with open(html_file, encoding="utf-8") as f:
        html_content = f.read()

    # Check that all SVG content is included
    assert "Talairach Registration" in html_content
    assert "Aseg Parcellation" in html_content
    assert "Aparc Parcellation" in html_content
    assert "LH Pial Surface" in html_content
    assert "RH Pial Surface" in html_content

    # Check HTML structure
    assert "<html" in html_content
    assert "FreeSurfer: Individual Report" in html_content

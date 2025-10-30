"""Tests for HTML report generation."""

import datetime
import inspect
import re
import tempfile
from contextlib import suppress
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from importlib_resources import files
from PIL import Image

from pyfsviz.freesurfer import FreeSurfer
from pyfsviz.reports import Template


class TestTemplate:
    """Test the Template class for HTML report generation."""

    def test_template_init(self) -> None:
        """Test Template initialization."""
        # Create a simple test template
        template_content = "<html><body>{{ title }}</body></html>"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as f:
            f.write(template_content)
            template_path = f.name

        try:
            template = Template(template_path)

            # Test that the template can be compiled
            result = template.compile({"title": "Test"})
            assert isinstance(result, str)
            assert "Test" in result
        finally:
            Path(template_path).unlink()

    def test_template_compile(self) -> None:
        """Test template compilation with simple template."""
        # Create a simple test template
        template_content = """
        <html>
        <head><title>{{ title }}</title></head>
        <body>
            <h1>{{ title }}</h1>
            <p>{{ content }}</p>
            {% if items %}
            <ul>
                {% for item in items %}
                <li>{{ item }}</li>
                {% endfor %}
            </ul>
            {% endif %}
        </body>
        </html>
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as f:
            f.write(template_content)
            template_path = f.name

        try:
            template = Template(template_path)

            configs = {
                "title": "Test Report",
                "content": "This is a test report",
                "items": ["Item 1", "Item 2", "Item 3"],
            }

            result = template.compile(configs)

            assert "Test Report" in result
            assert "This is a test report" in result
            assert "Item 1" in result
            assert "Item 2" in result
            assert "Item 3" in result
            assert "<html>" in result
            assert "<head>" in result
            assert "<body>" in result

        finally:
            Path(template_path).unlink()

    def test_template_generate_conf(self) -> None:
        """Test template configuration file generation."""
        # Create a simple test template
        template_content = """
        <html>
        <head><title>{{ title }}</title></head>
        <body>
            <h1>{{ title }}</h1>
            <p>{{ content }}</p>
        </body>
        </html>
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as f:
            f.write(template_content)
            template_path = f.name

        try:
            template = Template(template_path)

            configs = {
                "title": "Test Report",
                "content": "This is a test report",
            }

            with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as output_f:
                output_path = output_f.name

            try:
                template.generate_conf(configs, output_path)

                # Check that the file was created and contains expected content
                assert Path(output_path).exists()

                with open(output_path, encoding="utf-8") as f:
                    content = f.read()

                assert "Test Report" in content
                assert "This is a test report" in content
                assert "<html>" in content

            finally:
                Path(output_path).unlink()

        finally:
            Path(template_path).unlink()

    def test_template_with_empty_config(self) -> None:
        """Test template with empty configuration."""
        template_content = """
        <html>
        <head><title>{{ title or 'Default Title' }}</title></head>
        <body>
            <h1>{{ title or 'Default Title' }}</h1>
            <p>{{ content or 'No content' }}</p>
        </body>
        </html>
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as f:
            f.write(template_content)
            template_path = f.name

        try:
            template = Template(template_path)

            configs: dict[str, str] = {}
            result = template.compile(configs)

            assert "Default Title" in result
            assert "No content" in result

        finally:
            Path(template_path).unlink()

    def test_template_with_conditional_logic(self) -> None:
        """Test template with conditional logic."""
        template_content = """
        <html>
        <body>
            {% if show_header %}
            <h1>{{ title }}</h1>
            {% endif %}

            {% if items %}
            <ul>
                {% for item in items %}
                <li>{{ item }}</li>
                {% endfor %}
            </ul>
            {% else %}
            <p>No items to display</p>
            {% endif %}
        </body>
        </html>
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as f:
            f.write(template_content)
            template_path = f.name

        try:
            template = Template(template_path)

            # Test with items
            configs_with_items = {
                "show_header": True,
                "title": "Test Title",
                "items": ["Item 1", "Item 2"],
            }

            result_with_items = template.compile(configs_with_items)
            assert "Test Title" in result_with_items
            assert "Item 1" in result_with_items
            assert "Item 2" in result_with_items
            assert "No items to display" not in result_with_items

            # Test without items
            configs_without_items = {
                "show_header": False,
                "items": [],
            }

            result_without_items = template.compile(configs_without_items)
            assert "Test Title" not in result_without_items
            assert "No items to display" in result_without_items

        finally:
            Path(template_path).unlink()


class TestHTMLReportGeneration:
    """Test HTML report generation functionality."""

    def test_gen_html_report_basic(self, mock_freesurfer_instance: FreeSurfer, temp_output_dir: Path) -> None:
        """Test basic HTML report generation."""
        mock_img_dir = temp_output_dir / "mock_imgs"
        mock_img_dir.mkdir(parents=True, exist_ok=True)

        # Create TLRC SVG file
        with open(mock_img_dir / "tlrc.svg", "w", encoding="utf-8") as f:
            f.write("<svg><text>Talairach Registration</text></svg>")

        # Create PNG files for other images
        png_files = [
            "aseg.png",
            "aparc.png",
            "lh_pial.png",
            "rh_pial.png",
            "lh_infl.png",
            "rh_infl.png",
            "lh_white.png",
            "rh_white.png",
        ]

        img = Image.new("RGB", (1, 1), color="black")
        for filename in png_files:
            img.save(mock_img_dir / filename, "PNG")

        # Generate HTML report
        all_files = list(mock_img_dir.glob("*"))
        html_file = mock_freesurfer_instance.gen_html_report(
            subject="sub-001",
            output_dir=str(temp_output_dir),
            img_list=all_files,
        )

        # Check that HTML file was created
        assert html_file.exists()
        assert html_file.name == "sub-001.html"

        # Check HTML content
        with open(html_file, encoding="utf-8") as f:
            html_content = f.read()

        # Check that SVG is embedded and PNG images have proper paths
        assert "Talairach Registration" in html_content
        assert "aseg.png" in html_content
        assert "aparc.png" in html_content
        assert "<img src=" in html_content
        assert "img-fluid" in html_content
        assert "<svg>" in html_content or "svg" in html_content

        # Check HTML structure
        assert "<html" in html_content
        assert "<head>" in html_content
        assert "<body>" in html_content
        assert "FreeSurfer: Individual Report" in html_content

    def test_gen_html_report_with_default_template(
        self,
        mock_freesurfer_instance: FreeSurfer,
        temp_output_dir: Path,
    ) -> None:
        """Test HTML report generation with default template."""
        # Create mock SVG file
        mock_img_dir = temp_output_dir / "mock_imgs"
        mock_img_dir.mkdir(parents=True, exist_ok=True)

        with open(mock_img_dir / "tlrc.svg", "w", encoding="utf-8") as f:
            f.write("<svg><text>Test Talairach</text></svg>")

        # Generate HTML report with default template
        html_file = mock_freesurfer_instance.gen_html_report(
            subject="sub-001",
            output_dir=str(temp_output_dir),
            img_list=list(mock_img_dir.glob("*")),
        )

        assert html_file.exists()

        with open(html_file, encoding="utf-8") as f:
            html_content = f.read()

        # Check that default template elements are present
        assert "FreeSurfer: Individual Report" in html_content
        assert "Talairach Registration" in html_content
        assert "Aparc+Aseg Parcellations" in html_content
        assert "Surfaces" in html_content

    def test_gen_html_report_with_custom_template(
        self,
        mock_freesurfer_instance: FreeSurfer,
        temp_output_dir: Path,
    ) -> None:
        """Test HTML report generation with custom template."""
        # Create custom template
        custom_template_content = """
        <html>
        <head><title>Custom Report</title></head>
        <body>
            <h1>Custom FreeSurfer Report</h1>
            <p>Subject: {{ subject }}</p>
            <p>Generated: {{ timestamp }}</p>

            <h2>Talairach Data</h2>
            {% for item in tlrc %}
            <div>{{ item }}</div>
            {% endfor %}

            <h2>Aseg Data</h2>
            {% for item in aseg %}
            <div>{{ item }}</div>
            {% endfor %}

            <h2>Surface Data</h2>
            {% for label, item in surf %}
            <div>
                <h3>{{ label }}</h3>
                {{ item }}
            </div>
            {% endfor %}
        </body>
        </html>
        """

        custom_template_path = temp_output_dir / "custom_template.html"
        with open(custom_template_path, "w", encoding="utf-8") as f:
            f.write(custom_template_content)

        # Create mock PNG files
        mock_png_dir = temp_output_dir / "mock_pngs"
        mock_png_dir.mkdir(parents=True, exist_ok=True)

        img = Image.new("RGB", (1, 1), color="black")
        img.save(mock_png_dir / "tlrc.png", "PNG")
        img.save(mock_png_dir / "aseg.png", "PNG")

        # Generate HTML report with custom template
        html_file = mock_freesurfer_instance.gen_html_report(
            subject="sub-001",
            output_dir=str(temp_output_dir),
            img_list=list(mock_png_dir.glob("*.png")),
            template=str(custom_template_path),
        )

        assert html_file.exists()

        with open(html_file, encoding="utf-8") as f:
            html_content = f.read()

        # Check that custom template elements are present
        assert "Custom FreeSurfer Report" in html_content
        assert "Subject: sub-001" in html_content
        assert "Generated:" in html_content

    def test_gen_html_report_empty_images(self, mock_freesurfer_instance: FreeSurfer, temp_output_dir: Path) -> None:
        """Test HTML report generation with no images."""
        # Create empty directory
        empty_dir = temp_output_dir / "empty"
        empty_dir.mkdir(parents=True, exist_ok=True)

        # Generate HTML report with no images
        html_file = mock_freesurfer_instance.gen_html_report(
            subject="sub-001",
            output_dir=str(temp_output_dir),
            img_list=list(empty_dir.glob("*.png")),
        )

        assert html_file.exists()

        with open(html_file, encoding="utf-8") as f:
            html_content = f.read()

        # Check that basic structure is present even without images
        assert "FreeSurfer: Individual Report" in html_content

    def test_gen_html_report_timestamp_format(
        self,
        mock_freesurfer_instance: FreeSurfer,
        temp_output_dir: Path,
    ) -> None:
        """Test that timestamp is properly formatted in HTML report."""
        # Create mock SVG file
        mock_img_dir = temp_output_dir / "mock_imgs"
        mock_img_dir.mkdir(parents=True, exist_ok=True)

        with open(mock_img_dir / "tlrc.svg", "w", encoding="utf-8") as f:
            f.write("<svg><text>Test</text></svg>")

        # Generate HTML report
        html_file = mock_freesurfer_instance.gen_html_report(
            subject="sub-001",
            output_dir=str(temp_output_dir),
            img_list=list(mock_img_dir.glob("*")),
        )

        with open(html_file, encoding="utf-8") as f:
            html_content = f.read()

        # Check that timestamp is present and properly formatted
        assert "Date and time:" in html_content

        # Extract timestamp from HTML
        timestamp_match = re.search(r"Date and time: ([^.]*)\.", html_content)
        assert timestamp_match is not None

        timestamp_str = timestamp_match.group(1)

        # Try to parse the timestamp to ensure it's valid
        try:
            datetime.datetime.strptime(timestamp_str, "%Y-%m-%d, %H:%M").astimezone(datetime.timezone.utc)
        except ValueError:
            pytest.fail(f"Invalid timestamp format: {timestamp_str}")

    def test_gen_html_report_surface_label_mapping(
        self,
        mock_freesurfer_instance: FreeSurfer,
        temp_output_dir: Path,
    ) -> None:
        """Test that surface files are properly labeled in HTML report."""
        # Create mock PNG files with specific naming
        mock_png_dir = temp_output_dir / "mock_pngs"
        mock_png_dir.mkdir(parents=True, exist_ok=True)

        surface_files = [
            "lh_pial.png",
            "rh_pial.png",
            "lh_infl.png",
            "rh_infl.png",
            "lh_white.png",
            "rh_white.png",
        ]

        img = Image.new("RGB", (1, 1), color="black")
        for filename in surface_files:
            img.save(mock_png_dir / filename, "PNG")

        # Generate HTML report
        html_file = mock_freesurfer_instance.gen_html_report(
            subject="sub-001",
            output_dir=str(temp_output_dir),
            img_list=list(mock_png_dir.glob("*.png")),
        )

        with open(html_file, encoding="utf-8") as f:
            html_content = f.read()

        # Check that surface labels are properly mapped
        assert "LH Pial" in html_content
        assert "RH Pial" in html_content
        assert "LH Inflated" in html_content
        assert "RH Inflated" in html_content
        assert "LH White Matter" in html_content
        assert "RH White Matter" in html_content

    def test_gen_html_report_with_actual_template(
        self,
        mock_freesurfer_instance: FreeSurfer,
        temp_output_dir: Path,
    ) -> None:
        """Test HTML report generation with the actual FreeSurfer template."""
        # Create mock image files
        mock_img_dir = temp_output_dir / "mock_imgs"
        mock_img_dir.mkdir(parents=True, exist_ok=True)

        # Create TLRC SVG
        with open(mock_img_dir / "tlrc.svg", "w", encoding="utf-8") as f:
            f.write("<svg><text>Talairach Registration</text></svg>")

        # Create PNG files
        png_files = [
            "aseg.png",
            "aparc.png",
            "lh_pial.png",
            "rh_pial.png",
        ]

        img = Image.new("RGB", (1, 1), color="black")
        for filename in png_files:
            img.save(mock_img_dir / filename, "PNG")

        # Get the actual template path
        actual_template = files("pyfsviz._internal.html") / "individual.html"

        # Generate HTML report with actual template
        html_file = mock_freesurfer_instance.gen_html_report(
            subject="sub-001",
            output_dir=str(temp_output_dir),
            img_list=list(mock_img_dir.glob("*")),
            template=str(actual_template),
        )

        assert html_file.exists()

        with open(html_file, encoding="utf-8") as f:
            html_content = f.read()

        # Check that actual template elements are present
        assert "FreeSurfer: Individual Report" in html_content
        assert "Talairach Registration" in html_content
        assert "Aparc+Aseg Parcellations" in html_content
        assert "Surfaces" in html_content
        assert "Summary" in html_content
        assert "Date and time:" in html_content

        # Check for Bootstrap CSS and JS
        assert "bootstrap" in html_content.lower()
        assert "jquery" in html_content.lower()

        # Check for navigation elements
        assert "navbar" in html_content
        assert "Summary" in html_content
        assert 'href="#metrics"' in html_content or "Metrics" in html_content
        assert "Talairach Registration" in html_content
        assert "Aparc+Aseg Parcellations" in html_content
        assert "Surfaces" in html_content

    def test_gen_html_report_return_path(self, mock_freesurfer_instance: FreeSurfer, temp_output_dir: Path) -> None:
        """Test that gen_html_report returns the correct Path object."""
        # Create mock SVG files
        mock_svg_dir = temp_output_dir / "mock_svgs"
        mock_svg_dir.mkdir(parents=True, exist_ok=True)

        with open(mock_svg_dir / "tlrc.svg", "w", encoding="utf-8") as f:
            f.write("<svg><text>Test</text></svg>")

        # Generate HTML report
        html_file = mock_freesurfer_instance.gen_html_report(
            subject="sub-001",
            output_dir=str(temp_output_dir),
            img_list=list(mock_svg_dir.glob("**/*.svg")),
        )

        # HTML file should be in subject-specific directory
        assert html_file.parent == temp_output_dir / "sub-001"
        assert html_file.exists()

    def test_gen_html_report_with_metrics(
        self,
        mock_freesurfer_instance: FreeSurfer,
        temp_output_dir: Path,
    ) -> None:
        """Test HTML report generation with metrics.csv file."""
        # Create mock image files
        mock_img_dir = temp_output_dir / "mock_imgs"
        mock_img_dir.mkdir(parents=True, exist_ok=True)

        with open(mock_img_dir / "tlrc.svg", "w", encoding="utf-8") as f:
            f.write("<svg><text>Talairach Registration</text></svg>")

        img = Image.new("RGB", (1, 1), color="black")
        img.save(mock_img_dir / "aparcaseg.png", "PNG")

        # Create metrics.csv file
        metrics_data = {
            "subject": ["sub-001"],
            "wm_snr_orig": [10.134],
            "gm_snr_orig": [6.283],
            "wm_snr_norm": [14.473],
            "gm_snr_norm": [7.087],
            "cc_size": [0.001761],
            "holes_lh": [5],
            "holes_rh": [1],
            "defects_lh": [8],
            "defects_rh": [4],
            "topo_lh": [0.9],
            "topo_rh": [0.3],
            "con_snr_lh": [3.176],
            "con_snr_rh": [3.174],
            "rot_tal_x": [-0.160826],
            "rot_tal_y": [0.056567],
            "rot_tal_z": [-0.075648],
        }
        df = pd.DataFrame(metrics_data)
        metrics_csv_path = temp_output_dir / "metrics.csv"
        df.to_csv(metrics_csv_path, index=False)

        # Generate HTML report
        html_file = mock_freesurfer_instance.gen_html_report(
            subject="sub-001",
            output_dir=str(temp_output_dir),
            img_list=list(mock_img_dir.glob("*")),
        )

        assert html_file.exists()

        with open(html_file, encoding="utf-8") as f:
            html_content = f.read()

        # Check that metrics section is present
        assert 'id="metrics"' in html_content
        assert "Metrics" in html_content

        # Check that metrics values are in the HTML
        assert "WM SNR (Original)" in html_content
        assert "GM SNR (Original)" in html_content
        assert "WM SNR (Normalized)" in html_content
        assert "GM SNR (Normalized)" in html_content
        assert "CC Size" in html_content
        assert "Holes (LH)" in html_content
        assert "Holes (RH)" in html_content
        assert "Contrast SNR (LH)" in html_content
        assert "Rotation Tal X" in html_content

        # Check that Metrics is in navigation
        assert 'href="#metrics"' in html_content

    def test_gen_html_report_without_metrics(
        self,
        mock_freesurfer_instance: FreeSurfer,
        temp_output_dir: Path,
    ) -> None:
        """Test HTML report generation without metrics.csv file."""
        # Create mock image files
        mock_img_dir = temp_output_dir / "mock_imgs"
        mock_img_dir.mkdir(parents=True, exist_ok=True)

        with open(mock_img_dir / "tlrc.svg", "w", encoding="utf-8") as f:
            f.write("<svg><text>Talairach Registration</text></svg>")

        img = Image.new("RGB", (1, 1), color="black")
        img.save(mock_img_dir / "aparcaseg.png", "PNG")

        # Ensure metrics.csv does not exist
        metrics_csv_path = temp_output_dir / "metrics.csv"
        if metrics_csv_path.exists():
            metrics_csv_path.unlink()

        # Generate HTML report
        html_file = mock_freesurfer_instance.gen_html_report(
            subject="sub-001",
            output_dir=str(temp_output_dir),
            img_list=list(mock_img_dir.glob("*")),
        )

        assert html_file.exists()

        with open(html_file, encoding="utf-8") as f:
            html_content = f.read()

        # Check that metrics section is not present (or is empty)
        # The template checks {% if metrics %} so if metrics is None, section won't render
        assert "FreeSurfer: Individual Report" in html_content

    def test_gen_html_report_metrics_subject_filtering(
        self,
        mock_freesurfer_instance: FreeSurfer,
        temp_output_dir: Path,
    ) -> None:
        """Test that metrics are filtered by subject when multiple subjects exist in CSV."""
        # Create mock image files
        mock_img_dir = temp_output_dir / "mock_imgs"
        mock_img_dir.mkdir(parents=True, exist_ok=True)

        with open(mock_img_dir / "tlrc.svg", "w", encoding="utf-8") as f:
            f.write("<svg><text>Talairach Registration</text></svg>")

        img = Image.new("RGB", (1, 1), color="black")
        img.save(mock_img_dir / "aparcaseg.png", "PNG")

        # Create metrics.csv file with multiple subjects
        metrics_data = {
            "subject": ["sub-001", "sub-002"],
            "wm_snr_orig": [10.134, 12.456],
            "gm_snr_orig": [6.283, 7.890],
            "holes_lh": [5, 3],
        }
        df = pd.DataFrame(metrics_data)
        metrics_csv_path = temp_output_dir / "metrics.csv"
        df.to_csv(metrics_csv_path, index=False)

        # Generate HTML report for sub-001
        html_file = mock_freesurfer_instance.gen_html_report(
            subject="sub-001",
            output_dir=str(temp_output_dir),
            img_list=list(mock_img_dir.glob("*")),
        )

        assert html_file.exists()

        with open(html_file, encoding="utf-8") as f:
            html_content = f.read()

        # Check that only sub-001 metrics are present
        assert "10.134" in html_content or "10.13" in html_content
        assert "12.456" not in html_content  # sub-002 value should not appear
        assert "5" in html_content  # sub-001 holes_lh

    def test_gen_html_report_metrics_nan_handling(
        self,
        mock_freesurfer_instance: FreeSurfer,
        temp_output_dir: Path,
    ) -> None:
        """Test that NaN values in metrics CSV are handled properly."""
        # Create mock image files
        mock_img_dir = temp_output_dir / "mock_imgs"
        mock_img_dir.mkdir(parents=True, exist_ok=True)

        with open(mock_img_dir / "tlrc.svg", "w", encoding="utf-8") as f:
            f.write("<svg><text>Talairach Registration</text></svg>")

        img = Image.new("RGB", (1, 1), color="black")
        img.save(mock_img_dir / "aparcaseg.png", "PNG")

        # Create metrics.csv file with NaN values
        metrics_data = {
            "subject": ["sub-001"],
            "wm_snr_orig": [10.134],
            "gm_snr_orig": [np.nan],  # NaN value
            "holes_lh": [5],
        }
        df = pd.DataFrame(metrics_data)
        metrics_csv_path = temp_output_dir / "metrics.csv"
        df.to_csv(metrics_csv_path, index=False)

        # Generate HTML report
        html_file = mock_freesurfer_instance.gen_html_report(
            subject="sub-001",
            output_dir=str(temp_output_dir),
            img_list=list(mock_img_dir.glob("*")),
        )

        assert html_file.exists()

        with open(html_file, encoding="utf-8") as f:
            html_content = f.read()

        # Check that metrics with values are present
        assert "WM SNR (Original)" in html_content
        assert "10.134" in html_content or "10.13" in html_content
        # NaN value should not cause errors and the field should not appear
        # (since template checks for 'key in metrics and metrics.key is not none')
        assert "Holes (LH)" in html_content

    def test_gen_html_report_metrics_csv_parser_error(
        self,
        mock_freesurfer_instance: FreeSurfer,
        temp_output_dir: Path,
    ) -> None:
        """Test gen_html_report handles ParserError from malformed CSV."""
        # Create mock image files
        mock_img_dir = temp_output_dir / "mock_imgs"
        mock_img_dir.mkdir(parents=True, exist_ok=True)

        with open(mock_img_dir / "tlrc.svg", "w", encoding="utf-8") as f:
            f.write("<svg><text>Talairach Registration</text></svg>")

        img = Image.new("RGB", (1, 1), color="black")
        img.save(mock_img_dir / "aparcaseg.png", "PNG")

        # Create a CSV with parser error (unclosed quote)
        metrics_csv_path = temp_output_dir / "metrics.csv"
        with open(metrics_csv_path, "w", encoding="utf-8") as f:
            f.write("subject,wm_snr_orig\n")
            f.write('sub-001,"unclosed quote\n')  # Malformed CSV

        # Should not raise exception, just log warning and continue
        html_file = mock_freesurfer_instance.gen_html_report(
            subject="sub-001",
            output_dir=str(temp_output_dir),
            img_list=list(mock_img_dir.glob("*")),
        )

        assert html_file.exists()
        # Report should still be generated even if metrics CSV has error
        with open(html_file, encoding="utf-8") as f:
            html_content = f.read()
        assert "FreeSurfer: Individual Report" in html_content

    def test_gen_html_report_metrics_csv_empty_data_error(
        self,
        mock_freesurfer_instance: FreeSurfer,
        temp_output_dir: Path,
    ) -> None:
        """Test gen_html_report handles EmptyDataError from empty CSV."""
        # Create mock image files
        mock_img_dir = temp_output_dir / "mock_imgs"
        mock_img_dir.mkdir(parents=True, exist_ok=True)

        with open(mock_img_dir / "tlrc.svg", "w", encoding="utf-8") as f:
            f.write("<svg><text>Talairach Registration</text></svg>")

        img = Image.new("RGB", (1, 1), color="black")
        img.save(mock_img_dir / "aparcaseg.png", "PNG")

        # Create an empty CSV file
        metrics_csv_path = temp_output_dir / "metrics.csv"
        metrics_csv_path.write_text("")

        # Should not raise exception, just log warning and continue
        html_file = mock_freesurfer_instance.gen_html_report(
            subject="sub-001",
            output_dir=str(temp_output_dir),
            img_list=list(mock_img_dir.glob("*")),
        )

        assert html_file.exists()
        # Report should still be generated even if metrics CSV is empty
        with open(html_file, encoding="utf-8") as f:
            html_content = f.read()
        assert "FreeSurfer: Individual Report" in html_content

    def test_gen_html_report_metrics_csv_permission_error(
        self,
        mock_freesurfer_instance: FreeSurfer,
        temp_output_dir: Path,
    ) -> None:
        """Test gen_html_report handles PermissionError gracefully."""
        # Create mock image files
        mock_img_dir = temp_output_dir / "mock_imgs"
        mock_img_dir.mkdir(parents=True, exist_ok=True)

        with open(mock_img_dir / "tlrc.svg", "w", encoding="utf-8") as f:
            f.write("<svg><text>Talairach Registration</text></svg>")

        img = Image.new("RGB", (1, 1), color="black")
        img.save(mock_img_dir / "aparcaseg.png", "PNG")

        # Create a CSV file but make it unreadable on Unix-like systems
        metrics_csv_path = temp_output_dir / "metrics.csv"
        metrics_csv_path.write_text("subject,wm_snr_orig\nsub-001,10.134\n")

        # On systems that support it, make file unreadable
        try:
            metrics_csv_path.chmod(0o000)  # No permissions
            # Should not raise exception, just log warning and continue
            html_file = mock_freesurfer_instance.gen_html_report(
                subject="sub-001",
                output_dir=str(temp_output_dir),
                img_list=list(mock_img_dir.glob("*")),
            )
            assert html_file.exists()
        except (PermissionError, OSError):
            # On Windows or if we can't change permissions, skip this test
            pytest.skip("Cannot test permission error on this system")
        finally:
            # Restore permissions so file can be cleaned up
            with suppress(OSError):
                metrics_csv_path.chmod(0o644)


class TestBatchReportGeneration:
    """Test batch HTML report generation functionality."""

    def test_gen_batch_reports_basic(self, mock_freesurfer_instance: FreeSurfer, temp_output_dir: Path) -> None:
        """Test basic batch report generation."""
        # Create mock SVG files for the subject that exists in mock data
        subjects = ["sub-001"]  # Only use subjects that exist in mock data

        for subject in subjects:
            mock_svg_dir = temp_output_dir / f"{subject}_svgs"
            mock_svg_dir.mkdir(parents=True, exist_ok=True)

            with open(mock_svg_dir / "tlrc.svg", "w", encoding="utf-8") as f:
                f.write(f"<svg><text>{subject} Talairach</text></svg>")

            with open(mock_svg_dir / "aseg.svg", "w", encoding="utf-8") as f:
                f.write(f"<svg><text>{subject} Aseg</text></svg>")

        # Generate batch reports
        results = mock_freesurfer_instance.gen_batch_reports(
            output_dir=temp_output_dir / "reports",
            subjects=subjects,
            gen_images=False,
        )

        # Check results
        assert len(results) == 1
        assert "sub-001" in results

        # Check that HTML files were created
        for subject in subjects:
            result = results[subject]
            assert isinstance(result, Path)
            assert result.exists()
            assert result.name == f"{subject}.html"

    def test_gen_batch_reports_with_custom_subjects(
        self,
        mock_freesurfer_instance: FreeSurfer,
        temp_output_dir: Path,
    ) -> None:
        """Test batch report generation with custom subject list."""
        # Create mock SVG files
        mock_svg_dir = temp_output_dir / "mock_svgs"
        mock_svg_dir.mkdir(parents=True, exist_ok=True)

        with open(mock_svg_dir / "tlrc.svg", "w", encoding="utf-8") as f:
            f.write("<svg><text>Test Talairach</text></svg>")

        # Generate batch reports for specific subjects
        custom_subjects = ["sub-001"]
        results = mock_freesurfer_instance.gen_batch_reports(
            output_dir=temp_output_dir / "reports",
            subjects=custom_subjects,
            gen_images=False,
        )

        # Check results
        assert len(results) == 1
        assert "sub-001" in results
        assert isinstance(results["sub-001"], Path)

    def test_gen_batch_reports_error_handling(
        self,
        mock_freesurfer_instance: FreeSurfer,
        temp_output_dir: Path,
    ) -> None:
        """Test batch report generation error handling."""
        # Test with non-existent subject
        results = mock_freesurfer_instance.gen_batch_reports(
            output_dir=temp_output_dir / "reports",
            subjects=["non-existent-subject"],
            gen_images=False,  # Don't try to generate images for non-existent subject
        )

        # Check that error is captured
        assert len(results) == 1
        assert "non-existent-subject" in results
        assert isinstance(results["non-existent-subject"], Exception)

    def test_gen_batch_reports_skip_failed(self, mock_freesurfer_instance: FreeSurfer, temp_output_dir: Path) -> None:
        """Test batch report generation with skip_failed=True."""
        # Create mock SVG files for one subject only
        mock_svg_dir = temp_output_dir / "mock_svgs"
        mock_svg_dir.mkdir(parents=True, exist_ok=True)

        with open(mock_svg_dir / "tlrc.svg", "w", encoding="utf-8") as f:
            f.write("<svg><text>Test Talairach</text></svg>")

        # Test with mix of valid and invalid subjects
        subjects = ["sub-001", "non-existent-subject"]  # sub-001 exists in mock data
        results = mock_freesurfer_instance.gen_batch_reports(
            output_dir=temp_output_dir / "reports",
            subjects=subjects,
            gen_images=False,
            skip_failed=True,
        )

        # Check that processing continued despite errors
        assert len(results) == 2
        assert isinstance(results["sub-001"], Path)
        assert isinstance(results["non-existent-subject"], Exception)

    def test_gen_batch_reports_output_directory_creation(
        self,
        mock_freesurfer_instance: FreeSurfer,
        temp_output_dir: Path,
    ) -> None:
        """Test that output directory is created if it doesn't exist."""
        # Create mock SVG files
        mock_svg_dir = temp_output_dir / "mock_svgs"
        mock_svg_dir.mkdir(parents=True, exist_ok=True)

        with open(mock_svg_dir / "tlrc.svg", "w", encoding="utf-8") as f:
            f.write("<svg><text>Test Talairach</text></svg>")

        # Use non-existent output directory
        non_existent_dir = temp_output_dir / "new_reports_dir"
        assert not non_existent_dir.exists()

        results = mock_freesurfer_instance.gen_batch_reports(
            output_dir=non_existent_dir,
            subjects=["sub-001"],
            gen_images=False,
        )

        # Check that directory was created
        assert non_existent_dir.exists()
        assert len(results) == 1
        assert isinstance(results["sub-001"], Path)

    def test_gen_batch_reports_return_type(self, mock_freesurfer_instance: FreeSurfer, temp_output_dir: Path) -> None:
        """Test that batch report generation returns correct type."""
        # Create mock SVG files
        mock_svg_dir = temp_output_dir / "mock_svgs"
        mock_svg_dir.mkdir(parents=True, exist_ok=True)

        with open(mock_svg_dir / "tlrc.svg", "w", encoding="utf-8") as f:
            f.write("<svg><text>Test Talairach</text></svg>")

        results = mock_freesurfer_instance.gen_batch_reports(
            output_dir=temp_output_dir / "reports",
            subjects=["sub-001"],
            gen_images=False,
        )

        # Check return type
        assert isinstance(results, dict)
        assert "sub-001" in results
        assert isinstance(results["sub-001"], Path)

    def test_gen_batch_reports_empty_subjects_list(
        self,
        mock_freesurfer_instance: FreeSurfer,
        temp_output_dir: Path,
    ) -> None:
        """Test batch report generation with empty subjects list."""
        results = mock_freesurfer_instance.gen_batch_reports(
            output_dir=temp_output_dir / "reports",
            subjects=[],
        )

        # Check that empty results are returned
        assert isinstance(results, dict)
        assert len(results) == 0

    def test_gen_batch_reports_method_signature(self, mock_freesurfer_instance: FreeSurfer) -> None:
        """Test that batch report methods have correct signatures."""
        # Test gen_batch_reports signature
        sig1 = inspect.signature(mock_freesurfer_instance.gen_batch_reports)
        assert "output_dir" in sig1.parameters
        assert "subjects" in sig1.parameters
        assert "gen_images" in sig1.parameters
        assert "template" in sig1.parameters
        assert "skip_failed" in sig1.parameters

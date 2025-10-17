"""Tests for HTML report generation."""

import datetime
import inspect
import re
import tempfile
from pathlib import Path

import pytest
from importlib_resources import files

from pyfs.freesurfer import FreeSurfer
from pyfs.reports import Template


class TestTemplate:
    """Test the Template class for HTML report generation."""

    def test_template_init(self) -> None:
        """Test Template initialization."""
        template_str = "test_template.html"
        template = Template(template_str)

        assert template.template_str == template_str
        assert template.env is not None
        assert template.env.trim_blocks is True
        assert template.env.lstrip_blocks is True
        assert template.env.autoescape is True

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
            "lh_infl.svg": "<svg><text>LH Inflated Surface</text></svg>",
            "rh_infl.svg": "<svg><text>RH Inflated Surface</text></svg>",
            "lh_white.svg": "<svg><text>LH White Matter</text></svg>",
            "rh_white.svg": "<svg><text>RH White Matter</text></svg>",
        }

        for filename, content in svg_files.items():
            with open(mock_svg_dir / filename, "w", encoding="utf-8") as f:
                f.write(content)

        # Generate HTML report
        html_file = mock_freesurfer_instance.gen_html_report(
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
        assert "LH Inflated Surface" in html_content
        assert "RH Inflated Surface" in html_content
        assert "LH White Matter" in html_content
        assert "RH White Matter" in html_content

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
        # Create mock SVG files
        mock_svg_dir = temp_output_dir / "mock_svgs"
        mock_svg_dir.mkdir(parents=True, exist_ok=True)

        with open(mock_svg_dir / "tlrc.svg", "w", encoding="utf-8") as f:
            f.write("<svg><text>Test Talairach</text></svg>")

        # Generate HTML report with default template
        html_file = mock_freesurfer_instance.gen_html_report(
            subject="sub-001",
            output_dir=str(temp_output_dir),
            img_out=str(mock_svg_dir),
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

        # Create mock SVG files
        mock_svg_dir = temp_output_dir / "mock_svgs"
        mock_svg_dir.mkdir(parents=True, exist_ok=True)

        with open(mock_svg_dir / "tlrc.svg", "w", encoding="utf-8") as f:
            f.write("<svg><text>Custom Talairach</text></svg>")

        with open(mock_svg_dir / "aseg.svg", "w", encoding="utf-8") as f:
            f.write("<svg><text>Custom Aseg</text></svg>")

        # Generate HTML report with custom template
        html_file = mock_freesurfer_instance.gen_html_report(
            subject="sub-001",
            output_dir=str(temp_output_dir),
            img_out=str(mock_svg_dir),
            template=str(custom_template_path),
        )

        assert html_file.exists()

        with open(html_file, encoding="utf-8") as f:
            html_content = f.read()

        # Check that custom template elements are present
        assert "Custom FreeSurfer Report" in html_content
        assert "Subject: sub-001" in html_content
        assert "Custom Talairach" in html_content
        assert "Custom Aseg" in html_content
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
            img_out=str(empty_dir),
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
        # Create mock SVG files
        mock_svg_dir = temp_output_dir / "mock_svgs"
        mock_svg_dir.mkdir(parents=True, exist_ok=True)

        with open(mock_svg_dir / "tlrc.svg", "w", encoding="utf-8") as f:
            f.write("<svg><text>Test</text></svg>")

        # Generate HTML report
        html_file = mock_freesurfer_instance.gen_html_report(
            subject="sub-001",
            output_dir=str(temp_output_dir),
            img_out=str(mock_svg_dir),
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
        # Create mock SVG files with specific naming
        mock_svg_dir = temp_output_dir / "mock_svgs"
        mock_svg_dir.mkdir(parents=True, exist_ok=True)

        surface_files = {
            "lh_pial.svg": "<svg><text>LH Pial</text></svg>",
            "rh_pial.svg": "<svg><text>RH Pial</text></svg>",
            "lh_infl.svg": "<svg><text>LH Inflated</text></svg>",
            "rh_infl.svg": "<svg><text>RH Inflated</text></svg>",
            "lh_white.svg": "<svg><text>LH White</text></svg>",
            "rh_white.svg": "<svg><text>RH White</text></svg>",
        }

        for filename, content in surface_files.items():
            with open(mock_svg_dir / filename, "w", encoding="utf-8") as f:
                f.write(content)

        # Generate HTML report
        html_file = mock_freesurfer_instance.gen_html_report(
            subject="sub-001",
            output_dir=str(temp_output_dir),
            img_out=str(mock_svg_dir),
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
            with open(mock_svg_dir / filename, "w", encoding="utf-8") as f:
                f.write(content)

        # Get the actual template path
        actual_template = files("pyfs._internal.html") / "individual.html"

        # Generate HTML report with actual template
        html_file = mock_freesurfer_instance.gen_html_report(
            subject="sub-001",
            output_dir=str(temp_output_dir),
            img_out=str(mock_svg_dir),
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
            img_out=str(mock_svg_dir),
        )

        assert html_file.parent == temp_output_dir
        assert html_file.exists()


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

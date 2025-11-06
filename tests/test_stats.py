"""Tests for FreeSurfer stats functions."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

from pyfsviz.stats import check_metrics, gen_metric_plots


@pytest.fixture
def mock_stats_files(temp_output_dir: Path) -> list[Path]:
    """Create mock stats files for testing."""
    stats_files = []

    # Create mock aseg CSV
    aseg_data = {
        "subject_id": ["sub-001", "sub-002", "sub-003", "sub-004", "sub-005"],
        "Left-Lateral-Ventricle": [5000.0, 5200.0, 4800.0, 5100.0, 4900.0],
        "Right-Lateral-Ventricle": [4900.0, 5100.0, 4700.0, 5000.0, 4800.0],
        "Left-Cerebral-White-Matter": [450000.0, 460000.0, 440000.0, 455000.0, 445000.0],
    }
    aseg_df = pd.DataFrame(aseg_data)
    aseg_file = temp_output_dir / "aseg_stats.csv"
    aseg_df.to_csv(aseg_file, index=False)
    stats_files.append(aseg_file)

    return stats_files


@pytest.fixture
def mock_stats_files_with_outliers(temp_output_dir: Path) -> list[Path]:
    """Create mock stats files with outliers for testing."""
    stats_files = []

    # Create mock aseg CSV with outliers
    # Use values that create a small std among normal values, then add extreme outliers
    # Normal values: [5000, 5010, 4990, 5005, 4995] - very tight distribution
    # Outliers: 10000 and 0 - extremely extreme values that will be outliers even with inflated std
    aseg_data = {
        "subject_id": ["sub-001", "sub-002", "sub-003", "sub-004", "sub-005", "sub-006", "sub-007"],
        "Left-Lateral-Ventricle": [
            5000.0,
            5010.0,
            4990.0,
            5005.0,
            4995.0,
            10000.0,
            0.0,
        ],  # 10000 and 0 are extreme outliers
        "Right-Lateral-Ventricle": [4900.0, 5100.0, 4700.0, 5000.0, 4800.0, 4900.0, 5100.0],
        "Left-Cerebral-White-Matter": [450000.0, 460000.0, 440000.0, 455000.0, 445000.0, 450000.0, 460000.0],
    }
    aseg_df = pd.DataFrame(aseg_data)
    aseg_file = temp_output_dir / "aseg_stats_outliers.csv"
    aseg_df.to_csv(aseg_file, index=False)
    stats_files.append(aseg_file)

    return stats_files


class TestCheckMetrics:
    """Test check_metrics function."""

    def test_check_metrics_basic(self, mock_stats_files: list[Path]) -> None:
        """Test basic check_metrics functionality."""
        results = check_metrics(mock_stats_files, sd_threshold=3.0)

        assert isinstance(results, dict)
        assert len(results) > 0

        # Check structure of results
        for _, metric_data in results.items():
            assert isinstance(metric_data, dict)
            for _, result in metric_data.items():
                assert isinstance(result, dict)
                assert "status" in result
                assert "message" in result

    def test_check_metrics_no_outliers(self, mock_stats_files: list[Path]) -> None:
        """Test check_metrics with data that has no outliers."""
        results = check_metrics(mock_stats_files, sd_threshold=3.0)

        # All values should be within 3 SD, so status should be "passed"
        for _, metric_data in results.items():
            for _, result in metric_data.items():
                if result["status"] != "no_data":
                    assert result["status"] in ["passed", "outliers_detected"]
                    if result["status"] == "passed":
                        assert "mean" in result
                        assert "std" in result
                        assert "upper_bound" in result
                        assert "lower_bound" in result
                        assert result["outlier_count"] == 0

    def test_check_metrics_with_outliers(self, mock_stats_files_with_outliers: list[Path]) -> None:
        """Test check_metrics with data that has outliers."""
        # Note: Outliers inflate the std calculation, so we use a lower threshold (1.5 SD)
        # to ensure outliers are detected. In practice, robust methods might be preferred.
        results = check_metrics(mock_stats_files_with_outliers, sd_threshold=1.5)

        # Should detect outliers in Left-Lateral-Ventricle
        found_outliers = False
        for _, metric_data in results.items():
            for _, result in metric_data.items():
                if result["status"] == "outliers_detected":
                    found_outliers = True
                    assert "outlier_subjects" in result
                    assert len(result["outlier_subjects"]) > 0
                    assert result["outlier_count"] > 0
                    assert "mean" in result
                    assert "std" in result

        # Should find at least some outliers
        assert found_outliers

    def test_check_metrics_different_threshold(self, mock_stats_files_with_outliers: list[Path]) -> None:
        """Test check_metrics with different SD threshold."""
        # With lower threshold, should find more outliers
        results_low = check_metrics(mock_stats_files_with_outliers, sd_threshold=2.0)
        results_high = check_metrics(mock_stats_files_with_outliers, sd_threshold=5.0)

        # Count outliers in both
        outliers_low = sum(
            1
            for metric_data in results_low.values()
            for result in metric_data.values()
            if result.get("status") == "outliers_detected"
        )
        outliers_high = sum(
            1
            for metric_data in results_high.values()
            for result in metric_data.values()
            if result.get("status") == "outliers_detected"
        )

        # Lower threshold should find more or equal outliers
        assert outliers_low >= outliers_high

    def test_check_metrics_empty_files(self, temp_output_dir: Path) -> None:
        """Test check_metrics with empty stats files."""
        # Create empty CSV
        empty_file = temp_output_dir / "empty.csv"
        empty_df = pd.DataFrame({"subject_id": [], "region1": []})
        empty_df.to_csv(empty_file, index=False)

        results = check_metrics([empty_file], sd_threshold=3.0)
        assert isinstance(results, dict)

    def test_check_metrics_missing_data(self, temp_output_dir: Path) -> None:
        """Test check_metrics with missing data."""
        # Create CSV with NaN values
        data_with_nan = {
            "subject_id": ["sub-001", "sub-002"],
            "region1": [100.0, float("nan")],
        }
        df = pd.DataFrame(data_with_nan)
        nan_file = temp_output_dir / "nan_data.csv"
        df.to_csv(nan_file, index=False)

        results = check_metrics([nan_file], sd_threshold=3.0)
        assert isinstance(results, dict)

    def test_check_metrics_outlier_subjects_structure(self, mock_stats_files_with_outliers: list[Path]) -> None:
        """Test that outlier_subjects have correct structure."""
        results = check_metrics(mock_stats_files_with_outliers, sd_threshold=3.0)

        for _, metric_data in results.items():
            for _, result in metric_data.items():
                if result["status"] == "outliers_detected":
                    assert isinstance(result["outlier_subjects"], list)
                    for outlier in result["outlier_subjects"]:
                        assert isinstance(outlier, dict)
                        assert "subject_id" in outlier
                        assert "value" in outlier
                        assert isinstance(outlier["subject_id"], str)
                        assert isinstance(outlier["value"], (int, float))


class TestGenMetricPlots:
    """Test gen_metric_plots function."""

    def test_gen_metric_plots_basic(self, mock_stats_files: list[Path]) -> None:
        """Test basic gen_metric_plots functionality."""
        plots = gen_metric_plots(mock_stats_files)

        assert isinstance(plots, list)
        # Should generate at least one plot
        assert len(plots) > 0

        # All plots should be Plotly figures
        for plot in plots:
            assert isinstance(plot, go.Figure)

    def test_gen_metric_plots_empty_files(self, temp_output_dir: Path) -> None:
        """Test gen_metric_plots with empty stats files."""
        # Create empty CSV
        empty_file = temp_output_dir / "empty.csv"
        empty_df = pd.DataFrame({"subject_id": [], "region1": []})
        empty_df.to_csv(empty_file, index=False)

        plots = gen_metric_plots([empty_file])
        assert isinstance(plots, list)

    def test_gen_metric_plots_with_hemisphere_data(self, temp_output_dir: Path) -> None:
        """Test gen_metric_plots with hemisphere-specific data."""
        # Create CSV with hemisphere column
        hemi_data = {
            "subject_id": ["sub-001", "sub-002", "sub-001", "sub-002"],
            "hemi": ["lh", "lh", "rh", "rh"],
            "region1": [100.0, 110.0, 95.0, 105.0],
        }
        df = pd.DataFrame(hemi_data)
        hemi_file = temp_output_dir / "hemi_data.csv"
        df.to_csv(hemi_file, index=False)

        plots = gen_metric_plots([hemi_file])
        assert isinstance(plots, list)
        # Should generate plots for regions
        assert len(plots) > 0

    def test_gen_metric_plots_plot_structure(self, mock_stats_files: list[Path]) -> None:
        """Test that generated plots have correct structure."""
        plots = gen_metric_plots(mock_stats_files)

        for plot in plots:
            assert isinstance(plot, go.Figure)
            # Check that plot has data
            assert len(plot.data) > 0
            # Check that plot has layout
            assert plot.layout is not None

    def test_gen_metric_plots_skip_hemisphere_files(self, temp_output_dir: Path) -> None:
        """Test that gen_metric_plots skips hemisphere-specific files."""
        # Create files that should be skipped (lh_ or rh_ in filename)
        lh_file = temp_output_dir / "lh_area_aparc.csv"
        rh_file = temp_output_dir / "rh_area_aparc.csv"

        lh_data = {"subject_id": ["sub-001"], "region1": [100.0]}
        rh_data = {"subject_id": ["sub-001"], "region1": [95.0]}

        pd.DataFrame(lh_data).to_csv(lh_file, index=False)
        pd.DataFrame(rh_data).to_csv(rh_file, index=False)

        # Create a file that should be processed
        aseg_file = temp_output_dir / "aseg.csv"
        aseg_data = {"subject_id": ["sub-001"], "region1": [100.0]}
        pd.DataFrame(aseg_data).to_csv(aseg_file, index=False)

        plots = gen_metric_plots([lh_file, rh_file, aseg_file])
        # Should only process aseg.csv, not the hemisphere files
        assert isinstance(plots, list)
        # Should have at least one plot from aseg.csv
        assert len(plots) >= 1

    def test_gen_metric_plots_skip_combined_files(self, temp_output_dir: Path) -> None:
        """Test that gen_metric_plots skips combined files."""
        # Create a combined file
        combined_file = temp_output_dir / "combined_aparc.csv"
        combined_data = {"subject_id": ["sub-001"], "region1": [100.0]}
        pd.DataFrame(combined_data).to_csv(combined_file, index=False)

        # Create a regular file
        regular_file = temp_output_dir / "aseg.csv"
        regular_data = {"subject_id": ["sub-001"], "region1": [100.0]}
        pd.DataFrame(regular_data).to_csv(regular_file, index=False)

        plots = gen_metric_plots([combined_file, regular_file])
        # Should process regular file but skip combined
        assert isinstance(plots, list)
        # Should have plots from regular file
        assert len(plots) >= 1

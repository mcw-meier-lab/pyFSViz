"""Tests for FreeSurfer stats functions."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from pyfsviz.stats import (
    _add_synthseg_tiv_to_aseg,
    _comparison_metric_label,
    _get_aparc_stats,
    _get_aseg_stats,
    _read_synthseg_tiv,
    check_metrics,
    compare_group_metrics,
    gen_group_comparison_plots,
    gen_metric_plots,
    summarize_outlier_subjects,
)


@pytest.fixture
def mock_stats_files(temp_output_dir: Path) -> list[Path]:
    """Create mock stats files for testing."""
    stats_files = []

    # Create mock aseg CSV
    aseg_data = {
        "Measure:volume": ["sub-001", "sub-002", "sub-003", "sub-004", "sub-005"],
        "Left-Lateral-Ventricle": [5000.0, 5200.0, 4800.0, 5100.0, 4900.0],
        "Right-Lateral-Ventricle": [4900.0, 5100.0, 4700.0, 5000.0, 4800.0],
        "Left-Cerebral-White-Matter": [
            450000.0,
            460000.0,
            440000.0,
            455000.0,
            445000.0,
        ],
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
        "subject_id": [
            "sub-001",
            "sub-002",
            "sub-003",
            "sub-004",
            "sub-005",
            "sub-006",
            "sub-007",
        ],
        "Left-Lateral-Ventricle": [
            5000.0,
            5010.0,
            4990.0,
            5005.0,
            4995.0,
            10000.0,
            0.0,
        ],  # 10000 and 0 are extreme outliers
        "Right-Lateral-Ventricle": [
            4900.0,
            5100.0,
            4700.0,
            5000.0,
            4800.0,
            4900.0,
            5100.0,
        ],
        "Left-Cerebral-White-Matter": [
            450000.0,
            460000.0,
            440000.0,
            455000.0,
            445000.0,
            450000.0,
            460000.0,
        ],
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

    def test_check_metrics_with_outliers(
        self,
        mock_stats_files_with_outliers: list[Path],
    ) -> None:
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

    def test_check_metrics_different_threshold(
        self,
        mock_stats_files_with_outliers: list[Path],
    ) -> None:
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

    def test_check_metrics_outlier_subjects_structure(
        self,
        mock_stats_files_with_outliers: list[Path],
    ) -> None:
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

    def test_gen_metric_plots_plot_structure(
        self,
        mock_stats_files: list[Path],
    ) -> None:
        """Test that generated plots have correct structure."""
        plots = gen_metric_plots(mock_stats_files)

        for plot in plots:
            assert isinstance(plot, go.Figure)
            # Check that plot has data
            assert len(plot.data) > 0
            # Check that plot has layout
            assert plot.layout is not None

    def test_gen_metric_plots_includes_hemisphere_files(
        self,
        temp_output_dir: Path,
    ) -> None:
        """Hemisphere aparc tables are plotted, not skipped."""
        lh_file = temp_output_dir / "lh_area_aparc.csv"
        rh_file = temp_output_dir / "rh_area_aparc.csv"
        aseg_file = temp_output_dir / "aseg.csv"

        pd.DataFrame({"subject_id": ["sub-001"], "region1": [100.0]}).to_csv(
            lh_file,
            index=False,
        )
        pd.DataFrame({"subject_id": ["sub-001"], "region1": [95.0]}).to_csv(
            rh_file,
            index=False,
        )
        pd.DataFrame({"subject_id": ["sub-001"], "region1": [80.0]}).to_csv(
            aseg_file,
            index=False,
        )

        aseg_only = gen_metric_plots([aseg_file])
        plots = gen_metric_plots([lh_file, rh_file, aseg_file])

        assert len(plots) > len(aseg_only)
        metrics = {fig.layout.meta["metric"] for fig in plots}
        assert metrics == {"lh_area_aparc", "rh_area_aparc", "aseg"}


class TestSummarizeOutlierSubjects:
    """Test summarize_outlier_subjects function."""

    def test_summarize_outlier_subjects(
        self,
        mock_stats_files_with_outliers: list[Path],
    ) -> None:
        """Test outlier subjects are aggregated by subject ID."""
        quality_summary = check_metrics(
            mock_stats_files_with_outliers,
            sd_threshold=1.5,
        )
        summary = summarize_outlier_subjects(quality_summary)

        assert isinstance(summary, list)
        assert len(summary) > 0
        subject_ids = {item["subject_id"] for item in summary}
        assert {"sub-006", "sub-007"} & subject_ids
        assert summary[0]["outlier_count"] > 0
        assert isinstance(summary[0]["findings"], list)

    def test_summarize_outlier_subjects_none(
        self,
        mock_stats_files: list[Path],
    ) -> None:
        """Test empty summary when no outliers are present."""
        quality_summary = check_metrics(mock_stats_files, sd_threshold=3.0)
        summary = summarize_outlier_subjects(quality_summary)

        assert summary == []


class TestCompareGroupMetrics:
    """Test group comparison helpers."""

    @pytest.fixture
    def group_stats_file(self, temp_output_dir: Path) -> Path:
        """Create a mock aseg stats file with two groups of subjects."""
        data = {
            "subject_id": ["sub-001", "sub-002", "sub-003", "sub-004"],
            "Left-Lateral-Ventricle": [5000.0, 5100.0, 7000.0, 7100.0],
            "Right-Lateral-Ventricle": [4900.0, 5000.0, 6800.0, 6900.0],
        }
        stats_file = temp_output_dir / "aseg.csv"
        pd.DataFrame(data).to_csv(stats_file, index=False)
        return stats_file

    def test_compare_group_metrics(self, group_stats_file: Path) -> None:
        """Test between-group metric comparison returns stats for each group."""
        groups = {
            "control": ["sub-001", "sub-002"],
            "patient": ["sub-003", "sub-004"],
        }
        comparison = compare_group_metrics([group_stats_file], groups)

        assert "aseg" in comparison
        assert "Left-Lateral-Ventricle" in comparison["aseg"]
        assert comparison["aseg"]["Left-Lateral-Ventricle"]["control"]["n"] == 2
        assert comparison["aseg"]["Left-Lateral-Ventricle"]["patient"]["n"] == 2
        assert "comparison" not in comparison["aseg"]["Left-Lateral-Ventricle"]

    def test_compare_group_metrics_requires_two_groups(
        self,
        group_stats_file: Path,
    ) -> None:
        """Test that one group raises ValueError."""
        with pytest.raises(ValueError, match="At least two groups"):
            compare_group_metrics([group_stats_file], {"only": ["sub-001"]})

    def test_gen_group_comparison_plots(self, group_stats_file: Path) -> None:
        """Test group comparison plots are Plotly figures."""
        groups = {
            "control": ["sub-001", "sub-002"],
            "patient": ["sub-003", "sub-004"],
        }
        plots = gen_group_comparison_plots([group_stats_file], groups)

        assert isinstance(plots, list)
        assert len(plots) > 0
        assert isinstance(plots[0], go.Figure)
        assert plots[0].layout.meta["metric"] == "aseg"
        assert plots[0].layout.meta["label"] == "Aseg"
        html = plots[0].to_html(full_html=False, include_plotlyjs=False)
        assert "bdata" not in html
        assert "control" in html
        assert "sub-001" in html
        assert "5000" in html

    def test_comparison_metric_label_by_filename(self) -> None:
        """Each stats table gets its own tab label, including extra measures."""
        assert _comparison_metric_label("aseg") == "Aseg"
        assert _comparison_metric_label("lh_area_aparc") == "LH Area"
        assert _comparison_metric_label("rh_area_aparc") == "RH Area"
        assert _comparison_metric_label("lh_volume_aparc") == "LH Volume"
        assert _comparison_metric_label("rh_thickness_aparc") == "RH Thickness"
        assert _comparison_metric_label("lh_meancurv_aparc") == "LH Meancurv"


def _write_synthseg_csv(
    path: Path,
    *,
    subject: str,
    tiv: float,
    include_subject_column: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if include_subject_column:
        pd.DataFrame(
            {
                "subject": [subject],
                "total intracranial": [tiv],
                "left cerebral white matter": [400000.0],
            },
        ).to_csv(path, index=False)
    else:
        pd.DataFrame(
            {
                "total intracranial": [tiv],
                "left cerebral white matter": [400000.0],
            },
        ).to_csv(path, index=False)


def _touch_fs_stats(subjects_dir: Path, subject: str, filename: str) -> None:
    path = subjects_dir / subject / "stats" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# dummy FreeSurfer stats\n", encoding="utf-8")


def _write_aparc_table(
    path: Path,
    *,
    hemi: str,
    measure: str,
    subjects: list[str],
    regions: dict[str, list[float]],
) -> None:
    data: dict[str, list[str] | list[float]] = {f"{hemi}.aparc.{measure}": subjects}
    for region, values in regions.items():
        data[f"{hemi}_{region}_{measure}"] = values
    data["BrainSegVolNotVent"] = [1_100_000.0] * len(subjects)
    data["eTIV"] = [1_500_000.0] * len(subjects)
    pd.DataFrame(data).to_csv(path, index=False)


class TestSynthSegTIV:
    """Test merging SynthSeg total intracranial volume into aseg tables."""

    def test_read_synthseg_tiv_with_subject_column(self, temp_output_dir: Path) -> None:
        """Test reading TIV from a per-subject SynthSeg CSV."""
        _write_synthseg_csv(
            temp_output_dir / "sub-001" / "stats" / "synthseg.vol.csv",
            subject="sub-001",
            tiv=1500123.4,
        )

        assert _read_synthseg_tiv("sub-001", temp_output_dir) == pytest.approx(
            1500123.4,
        )

    def test_read_synthseg_tiv_without_subject_column(
        self,
        temp_output_dir: Path,
    ) -> None:
        """Test reading TIV when the CSV has only volume columns."""
        _write_synthseg_csv(
            temp_output_dir / "sub-001" / "stats" / "synthseg.vol.csv",
            subject="sub-001",
            tiv=1234567.0,
            include_subject_column=False,
        )

        assert _read_synthseg_tiv("sub-001", temp_output_dir) == pytest.approx(
            1234567.0,
        )

    def test_read_synthseg_tiv_missing_file(self, temp_output_dir: Path) -> None:
        """Test that a missing SynthSeg CSV returns None."""
        assert _read_synthseg_tiv("sub-001", temp_output_dir) is None

    def test_add_synthseg_tiv_to_aseg(self, temp_output_dir: Path) -> None:
        """Test that TIV is inserted into aseg.csv and aligned by subject."""
        for subject, tiv in (("sub-001", 1500000.0), ("sub-002", 1600000.0)):
            _write_synthseg_csv(
                temp_output_dir / subject / "stats" / "synthseg.vol.csv",
                subject=subject,
                tiv=tiv,
            )

        aseg_file = temp_output_dir / "aseg.csv"
        pd.DataFrame(
            {
                "Measure:volume": ["sub-001", "sub-002", "sub-003"],
                "Left-Lateral-Ventricle": [5000.0, 5200.0, 4800.0],
            },
        ).to_csv(aseg_file, index=False)

        result = _add_synthseg_tiv_to_aseg(
            aseg_file,
            ["sub-001", "sub-002", "sub-003"],
            temp_output_dir,
        )
        df = pd.read_csv(result)

        assert list(df.columns[:2]) == ["Measure:volume", "total intracranial"]
        assert df.loc[df["Measure:volume"] == "sub-001", "total intracranial"].iloc[0] == pytest.approx(1500000.0)
        assert df.loc[df["Measure:volume"] == "sub-002", "total intracranial"].iloc[0] == pytest.approx(1600000.0)
        assert pd.isna(df.loc[df["Measure:volume"] == "sub-003", "total intracranial"].iloc[0])

    def test_add_synthseg_tiv_does_not_use_etiv(self, temp_output_dir: Path) -> None:
        """Do not copy EstimatedTotalIntraCranialVol into the SynthSeg TIV column."""
        aseg_file = temp_output_dir / "aseg.csv"
        pd.DataFrame(
            {
                "Measure:volume": ["sub-001"],
                "Left-Lateral-Ventricle": [5000.0],
                "EstimatedTotalIntraCranialVol": [1500000.0],
                "total intracranial": [None],
            },
        ).to_csv(aseg_file, index=False)

        result = _add_synthseg_tiv_to_aseg(aseg_file, ["sub-001"], temp_output_dir)
        df = pd.read_csv(result)

        assert np.isnan(df.loc[0, "total intracranial"])
        assert df.loc[0, "EstimatedTotalIntraCranialVol"] == pytest.approx(1500000.0)

    def test_read_synthseg_tiv_tab_separated(self, temp_output_dir: Path) -> None:
        """Read SynthSeg TIV from a tab-delimited volume file."""
        path = temp_output_dir / "sub-001" / "stats" / "synthseg.vol.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "subject\ttotal intracranial\nsub-001\t1500123.4\n",
            encoding="utf-8",
        )

        assert _read_synthseg_tiv("sub-001", temp_output_dir) == pytest.approx(
            1500123.4,
        )

    def test_add_synthseg_tiv_skips_missing_aseg(self, temp_output_dir: Path) -> None:
        """Do not invent an ID-only aseg table when the real file is missing."""
        aseg_file = temp_output_dir / "reports" / "aseg.csv"
        assert not aseg_file.exists()

        result = _add_synthseg_tiv_to_aseg(aseg_file, ["sub-001"], temp_output_dir)

        assert result == aseg_file
        assert not aseg_file.exists()

    def test_get_aseg_stats_reuses_existing_table(
        self,
        temp_output_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Skip asegstats2table when group reports have already written aseg.csv."""
        monkeypatch.setenv("SUBJECTS_DIR", str(temp_output_dir))
        _write_synthseg_csv(
            temp_output_dir / "sub-001" / "stats" / "synthseg.vol.csv",
            subject="sub-001",
            tiv=1500000.0,
        )
        aseg_file = temp_output_dir / "aseg.csv"
        pd.DataFrame(
            {
                "Measure:volume": ["sub-001"],
                "Left-Lateral-Ventricle": [5000.0],
            },
        ).to_csv(aseg_file, index=False)

        with patch("pyfsviz.stats.AsegStats") as mock_aseg:
            result = _get_aseg_stats(
                ["sub-001"],
                "aseg.csv",
                output_dir=str(temp_output_dir),
            )

        mock_aseg.assert_not_called()
        df = pd.read_csv(result)
        assert result == aseg_file
        assert list(df.columns[:2]) == ["Measure:volume", "total intracranial"]
        assert df.loc[0, "total intracranial"] == pytest.approx(1500000.0)

    def test_get_aseg_stats_regenerates_stub_table(
        self,
        temp_output_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Re-run asegstats2table when an existing table has no region columns."""
        monkeypatch.setenv("SUBJECTS_DIR", str(temp_output_dir))
        _touch_fs_stats(temp_output_dir, "sub-001", "aseg.stats")
        aseg_file = temp_output_dir / "aseg.csv"
        pd.DataFrame({"Measure:volume": ["sub-001"]}).to_csv(aseg_file, index=False)

        def _write_real_table(*_args: object, **_kwargs: object) -> dict:
            pd.DataFrame(
                {
                    "Measure:volume": ["sub-001"],
                    "Left-Lateral-Ventricle": [5000.0],
                },
            ).to_csv(aseg_file, index=False)
            return {}

        with patch("pyfsviz.stats.AsegStats") as mock_aseg:
            mock_aseg.return_value.run.side_effect = _write_real_table
            mock_aseg.return_value._list_outputs.return_value = {
                "out_table": str(aseg_file),
            }
            result = _get_aseg_stats(
                ["sub-001"],
                "aseg.csv",
                output_dir=str(temp_output_dir),
            )

        mock_aseg.assert_called_once()
        df = pd.read_csv(result)
        assert "Left-Lateral-Ventricle" in df.columns
        assert df.loc[0, "Left-Lateral-Ventricle"] == pytest.approx(5000.0)

    def test_get_aseg_stats_does_not_invent_table_when_command_writes_nothing(
        self,
        temp_output_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If asegstats2table writes nothing, do not replace it with an ID-only stub."""
        monkeypatch.setenv("SUBJECTS_DIR", str(temp_output_dir))
        _touch_fs_stats(temp_output_dir, "sub-001", "aseg.stats")
        with patch("pyfsviz.stats.AsegStats") as mock_aseg:
            mock_aseg.return_value.run.return_value = {}
            mock_aseg.return_value._list_outputs.return_value = {
                "out_table": "aseg.csv",
            }
            result = _get_aseg_stats(
                ["sub-001"],
                "aseg.csv",
                output_dir=str(temp_output_dir),
            )

        mock_aseg.assert_called_once()
        assert result == temp_output_dir / "aseg.csv"
        assert not result.exists()

    def test_get_aparc_stats_continues_when_command_fails(
        self,
        temp_output_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A FreeSurfer IndexError from an empty aparc table does not abort the report."""
        monkeypatch.setenv("SUBJECTS_DIR", str(temp_output_dir))
        _touch_fs_stats(temp_output_dir, "sub-001", "lh.aparc.stats")
        with patch("pyfsviz.stats.AparcStats") as mock_aparc:
            mock_aparc.return_value.run.side_effect = RuntimeError(
                "IndexError: list index out of range",
            )
            results = _get_aparc_stats(
                ["sub-001"],
                "aparc.csv",
                output_dir=str(temp_output_dir),
                hemis=["lh"],
                measures=["area"],
            )

        mock_aparc.assert_called_once()
        assert results == []

    def test_get_aparc_stats_does_not_invent_tables_when_command_writes_nothing(
        self,
        temp_output_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If aparcstats2table writes nothing, do not replace it with ID-only stubs."""
        monkeypatch.setenv("SUBJECTS_DIR", str(temp_output_dir))
        for hemi in ("lh", "rh"):
            _touch_fs_stats(temp_output_dir, "sub-001", f"{hemi}.aparc.stats")
        with patch("pyfsviz.stats.AparcStats") as mock_aparc:
            mock_aparc.return_value.run.return_value = {}
            mock_aparc.return_value._list_outputs.return_value = {
                "out_table": "lh_area_aparc.csv",
            }
            results = _get_aparc_stats(
                ["sub-001"],
                "aparc.csv",
                output_dir=str(temp_output_dir),
            )

        assert mock_aparc.call_count == 6
        lh_area = temp_output_dir / "lh_area_aparc.csv"
        assert not lh_area.exists()
        assert results == []

    def test_get_aparc_stats_reuses_existing_tables(
        self,
        temp_output_dir: Path,
    ) -> None:
        """Skip aparcstats2table when group reports have already written aparc CSVs."""
        for measure in ("area", "volume", "thickness"):
            for hemi in ("lh", "rh"):
                pd.DataFrame(
                    {
                        f"{hemi}.aparc.{measure}": ["sub-001"],
                        f"{hemi}_bankssts_{measure}": [1.0],
                    },
                ).to_csv(temp_output_dir / f"{hemi}_{measure}_aparc.csv", index=False)

        with patch("pyfsviz.stats.AparcStats") as mock_aparc:
            results = _get_aparc_stats(
                ["sub-001"],
                "aparc.csv",
                output_dir=str(temp_output_dir),
            )

        mock_aparc.assert_not_called()
        df = pd.read_csv(temp_output_dir / "lh_area_aparc.csv")
        assert list(df.columns[:2]) == ["lh.aparc.area", "lh_bankssts_area"]
        assert any(path.name == "lh_area_aparc.csv" for path in results)

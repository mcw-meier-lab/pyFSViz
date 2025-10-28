"""Configuration for the pytest test suite."""

import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from pyfsviz.freesurfer import FreeSurfer
from tests.test_data_generator import setup_mock_freesurfer_environment


@pytest.fixture(scope="session")
def mock_freesurfer_env() -> Generator[tuple[Path, Path], None, None]:
    """Create a mock FreeSurfer environment for testing."""
    yield from setup_mock_freesurfer_environment()


@pytest.fixture(scope="session")
def mock_freesurfer_home(mock_freesurfer_env: tuple[Path, Path]) -> Path:
    """Get the mock FreeSurfer home directory."""
    freesurfer_home, _ = mock_freesurfer_env
    return freesurfer_home


@pytest.fixture(scope="session")
def mock_subjects_dir(mock_freesurfer_env: tuple[Path, Path]) -> Path:
    """Get the mock subjects directory."""
    _, subjects_dir = mock_freesurfer_env
    return subjects_dir


@pytest.fixture(scope="session")
def mock_freesurfer_instance(mock_freesurfer_home: Path, mock_subjects_dir: Path) -> Generator[FreeSurfer, None, None]:
    """Create a FreeSurfer instance with mock data."""
    # Temporarily set environment variables
    original_freesurfer_home = os.environ.get("FREESURFER_HOME")
    original_subjects_dir = os.environ.get("SUBJECTS_DIR")

    os.environ["FREESURFER_HOME"] = str(mock_freesurfer_home)
    os.environ["SUBJECTS_DIR"] = str(mock_subjects_dir)

    try:
        yield FreeSurfer()
    finally:
        # Restore original environment variables
        if original_freesurfer_home is not None:
            os.environ["FREESURFER_HOME"] = original_freesurfer_home
        elif "FREESURFER_HOME" in os.environ:
            del os.environ["FREESURFER_HOME"]

        if original_subjects_dir is not None:
            os.environ["SUBJECTS_DIR"] = original_subjects_dir
        elif "SUBJECTS_DIR" in os.environ:
            del os.environ["SUBJECTS_DIR"]


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

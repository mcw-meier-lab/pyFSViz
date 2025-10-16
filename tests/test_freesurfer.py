import os

import pytest
from matplotlib import colors

from pyfs.freesurfer import FreeSurfer, get_freesurfer_colormap


def test_get_colormap() -> None:
    colormap = get_freesurfer_colormap(os.environ["FREESURFER_HOME"])
    assert isinstance(colormap, colors.ListedColormap)


@pytest.fixture(scope="module")
def freesurfer() -> FreeSurfer:
    return FreeSurfer()


def test_get_subjects(freesurfer: FreeSurfer) -> None:
    subjects = freesurfer.get_subjects()
    assert len(subjects) > 0
    assert all(isinstance(subject, str) for subject in subjects)

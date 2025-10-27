"""pyFS package.

Python tools for FreeSurfer visualization and QA
"""

from __future__ import annotations

from pyfs import freesurfer, reports
from pyfs._internal.cli import get_parser, main
from pyfs.freesurfer import (
    FreeSurfer,
    get_freesurfer_colormap,
)
from pyfs.reports import (
    Template,
)

__all__: list[str] = [
    "FreeSurfer",
    "Template",
    "freesurfer",
    "get_freesurfer_colormap",
    "get_parser",
    "main",
    "reports",
]

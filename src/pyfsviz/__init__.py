"""pyFSViz package.

Python tools for FreeSurfer visualization and QA
"""

from __future__ import annotations

from pyfsviz._internal.cli import get_parser, main
from pyfsviz.freesurfer import (
    FreeSurfer,
    get_freesurfer_colormap,
)
from pyfsviz.reports import (
    Template,
)

__all__: list[str] = [
    "FreeSurfer",
    "Template",
    "get_freesurfer_colormap",
    "get_parser",
    "main",
]

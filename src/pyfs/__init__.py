"""pyFS package.

Python tools for FreeSurfer visualization and QA
"""

from __future__ import annotations

from pyfs._internal.cli import get_parser, main

__all__: list[str] = ["get_parser", "main"]

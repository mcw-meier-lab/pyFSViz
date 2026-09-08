from __future__ import annotations

import argparse
import sys
from typing import Any

from pyfsviz._internal import debug


class _DebugInfo(argparse.Action):
    def __init__(self, nargs: int | str | None = 0, **kwargs: Any) -> None:
        super().__init__(nargs=nargs, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        debug._print_debug_info()
        sys.exit(0)


def get_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser.

    Returns
    -------
    An argparse parser.
    """
    parser = argparse.ArgumentParser(prog="pyfsviz")
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {debug._get_version()}")
    parser.add_argument("--debug-info", action=_DebugInfo, help="Print debug information.")
    return parser


def main(args: list[str] | None = None) -> int:
    """Run the main program.

    This function is executed when you type `pyfsviz` or `python -m pyfsviz`.

    Parameters
    ----------
    args: Arguments passed from the command line.

    Returns
    -------
    An exit code.
    """
    parser = get_parser()
    opts = parser.parse_args(args=args)
    print(opts)
    return 0

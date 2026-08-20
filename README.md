# pyFSViz

[![ci](https://github.com/mcw-meier-lab/pyFSViz/workflows/ci/badge.svg)](https://github.com/mcw-meier-lab/pyFSViz/actions?query=workflow%3Aci)
[![documentation](https://img.shields.io/badge/docs-mkdocs-708FCC.svg?style=flat)](https://mcw-meier-lab.github.io/pyFSViz/)
[![pypi version](https://img.shields.io/pypi/v/pyfsviz.svg)](https://pypi.org/project/pyfsviz/)

Python tools for FreeSurfer visualization and QA. pyFSViz builds HTML reports
from existing `recon-all` output (individual screenshots and group metric
summaries). It does not run FreeSurfer reconstructions.

> **FreeSurfer and FSL are not part of this install.** Install those tools
> separately and source their setup scripts in the same shell or job before
> using pyFSViz. See
> [Prerequisites](https://mcw-meier-lab.github.io/pyFSViz/user-guide/prerequisites/).

## Installation

Requires Python 3.10 or newer. This installs the Python package only.

```bash
pip install pyfsviz
```

With [`uv`](https://docs.astral.sh/uv/):

```bash
uv tool install pyfsviz
```

## Usage

After [FreeSurfer and FSL are initialized](https://mcw-meier-lab.github.io/pyFSViz/user-guide/prerequisites/):

```python
from pyfsviz import FreeSurfer

fs = FreeSurfer()
fs.gen_batch_reports("reports/")
fs.gen_group_report("reports/")
```

See the [quick start](https://mcw-meier-lab.github.io/pyFSViz/user-guide/quickstart/)
for output layout, group comparisons, and batch flags.

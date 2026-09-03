# pyFSViz

[![ci](https://github.com/mcw-meier-lab/pyFSViz/workflows/ci/badge.svg)](https://github.com/mcw-meier-lab/pyFSViz/actions?query=workflow%3Aci)
[![documentation](https://img.shields.io/badge/docs-mkdocs-708FCC.svg?style=flat)](https://mcw-meier-lab.github.io/pyFSViz/)
[![pypi version](https://img.shields.io/pypi/v/pyfsviz.svg)](https://pypi.org/project/pyfsviz/)

## Description

Python tools for FreeSurfer visualization and quality assurance. pyFSViz builds HTML reports
from existing `recon-all` output (individual screenshots and group metric
summaries). It does not run FreeSurfer reconstructions.

While manual QA of FreeSurfer data is always recommended, this workflow become untenable with larger datasets, especially longitudinal ones. Several tools are available (some of which are utilized by this package), however, after doing manual QA myself for years, I've found a few key pieces missing. 
- The ability to check Talairach registrations (an important first step as this affects downstream processing and brain volume calculations)
- An easy way to check for outliers at a glance (can be done statistically later on but also useful in catching errors earlier on in the process, especially for large datasets)
- Comparison between groups (especially helpful for multi-site data that may have different scanner characteristics)

pyFSViz relies on code from Deep-MI's [fsqc](https://github.com/Deep-MI/fsqc/tree/stable), as well as other neuroimaging python packages such as [nipype](https://github.com/nipy/nipype) and [nireports](https://github.com/nipreps/nireports).

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

## Acknowledgements
Thanks to [pawamoy's copier project](https://github.com/pawamoy/copier-uv) for package templates!

## AI Disclosure
Please note that the default Cursor Agent was used in this project. This included models suchs as Grok-4.6 and Composer-2.5. The agent was used to improve base code, add testing, help with html development, and documentation.
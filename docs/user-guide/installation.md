---
title: Installation
---

# Installation

It's recommended to install pyFSViz through PyPI. Please note, this does **not** install FreeSurfer or
FSL; those must be set up separately. See [Prerequisites](prerequisites.md).

Requires Python 3.10 or newer.

```bash
pip install pyfsviz
```

With [`uv`](https://docs.astral.sh/uv/):

```bash
uv tool install pyfsviz
```

For a project-local environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pyfsviz
```

Use the same shell (or job script) for the virtualenv **and** the FreeSurfer /
FSL setup scripts. nipype finds `mri_convert` and `flirt` on `PATH`; a venv
does not provide those binaries.

After install, continue with [Quick start](quickstart.md).

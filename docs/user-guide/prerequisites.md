---
title: Prerequisites
---

# Prerequisites

pyFSViz builds QA reports from **existing** FreeSurfer reconstructions. It does
not run `recon-all`, and it does not install or initialize the neuroimaging
tools it calls.

!!! danger "Install and source FreeSurfer and FSL yourself"

    **This package does not ship, install, configure, or source FreeSurfer or
    FSL.** `pip install pyfsviz` only installs Python dependencies.

    Install those tools from their vendors, then initialize them in the **same
    shell or batch job** before you import pyFSViz. A Python virtualenv is not
    a substitute for sourcing the vendor setup scripts.

| Tool | Used for | What “initialized” means |
| --- | --- | --- |
| [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall) | Subject directories, color LUT, `mri_convert`, `asegstats2table`, `aparcstats2table` | `FREESURFER_HOME` and `SUBJECTS_DIR` are set, and FreeSurfer binaries are on `PATH` |
| [FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/index) | `flirt` for Talairach / MNI overlay images in individual reports | `FSLDIR` is set and `flirt` is on `PATH` |

Follow the official install guides linked above. Paths and setup-script names
differ by version and site; the commands below are examples only.

## FreeSurfer

1. Install FreeSurfer using the [vendor instructions](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall).
2. In each session (or job script), source the setup script so the environment
   is active:

    ```bash
    export FREESURFER_HOME=/path/to/freesurfer
    source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
    ```

    Some installs use `FreeSurferEnv.sh` instead of `SetUpFreeSurfer.sh`.

3. Point `SUBJECTS_DIR` at **your** recon-all output, not the default subjects
   tree bundled with FreeSurfer:

    ```bash
    export SUBJECTS_DIR=/path/to/your/subjects
    ```

pyFSViz reads `FREESURFER_HOME` and `SUBJECTS_DIR` when you construct
`FreeSurfer()` with no arguments. You can pass the same paths explicitly:

```python
from pyfsviz import FreeSurfer

fs = FreeSurfer(
    freesurfer_home="/path/to/freesurfer",
    subjects_dir="/path/to/your/subjects",
)
```

Passing those paths does **not** put `mri_convert` or the stats table commands
on `PATH`. The setup script (or an equivalent `module load`) still needs to
have run in that process environment.

## FSL

Individual reports that include Talairach registration images call FSL `flirt`
through nipype. Group-only stats reports do not require FSL.

1. Install FSL using the [vendor instructions](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/index).
2. Source FSL in the same session:

    ```bash
    export FSLDIR=/path/to/fsl
    source "$FSLDIR/etc/fslconf/fsl.sh"
    ```

Unlike FreeSurfer, pyFSViz has no constructor argument for FSL. nipype locates
`flirt` from the environment (`FSLDIR` and `PATH`).

## Check the environment

Run these in the same shell you will use for pyFSViz to verify your environment:

```bash
echo "$FREESURFER_HOME"
echo "$SUBJECTS_DIR"
echo "$FSLDIR"
command -v mri_convert
command -v asegstats2table
command -v flirt
```

Each variable should print a real directory, and each `command -v` should print
a binary path. If any are empty, the corresponding tool is not initialized in
this session.

On a cluster, `module load freesurfer` / `module load fsl` is only enough when
the module also sources the vendor scripts (or you source them afterwards in
the job). Initialize both tools in the submission script, not only in an
interactive login shell.

## Expected FreeSurfer subjects

pyFSViz does not create reconstructions. Each subject directory under
`SUBJECTS_DIR` should already contain a finished `recon-all` tree, including:

- `scripts/recon-all.log`
- `mri/` (`orig.mgz`, `wm.mgz`, `aparc+aseg.mgz`, …)
- `mri/transforms/talairach.lta` and `talairach.xfm.lta`
- `surf/` and `label/`
- `stats/` (and `stats/synthseg.vol.csv` when SynthSeg TIV should appear in
  group `aseg` tables)

`FreeSurfer.get_subjects()` only lists folders that contain
`mri/transforms/talairach.lta`. Batch reports warn when `recon-all.log` does
not end with `finished without error`.

FreeSurfer 7 and 8 reconstructions are both used. The package already handles
the extra LUT column in FreeSurfer 8+ and optional SynthSeg intracranial
volume in group tables.

Continue with [Installation](installation.md) and [Quick start](quickstart.md).

---
title: Troubleshooting
---

# Troubleshooting

Most failures mean FreeSurfer or FSL is missing from the **current** process
environment, or `SUBJECTS_DIR` does not contain finished reconstructions.
Confirm the tools are sourced as described in [Prerequisites](prerequisites.md).

| Symptom | Likely cause |
| --- | --- |
| `FREESURFER_HOME not found` | FreeSurfer was never sourced, or `freesurfer_home=` points at a missing directory |
| `SUBJECTS_DIR not found` | Env var unset / empty, or `subjects_dir=` is wrong |
| Empty `get_subjects()` list | No folders with `mri/transforms/talairach.lta` under the subjects directory |
| `mri_convert` / `asegstats2table` / `aparcstats2table` not found | FreeSurfer `bin` is not on `PATH` (passing paths to `FreeSurfer()` does not fix this) |
| nipype / `flirt` error during Talairach images | FSL not installed or not sourced (`FSLDIR`, `flirt` on `PATH`) |
| Batch warning that recon-all did not complete | `scripts/recon-all.log` does not end with `finished without error` |
| Group report missing SynthSeg TIV | No `stats/synthseg.vol.csv`, or no `total intracranial` column in that file |
| `skip_existing` left a subject unchanged | `{output_dir}/{subject}/{subject}.html` already exists |

## Check the environment

```bash
echo "$FREESURFER_HOME"
echo "$SUBJECTS_DIR"
echo "$FSLDIR"
command -v mri_convert
command -v asegstats2table
command -v flirt
python -c "from pyfsviz import FreeSurfer; print(FreeSurfer().get_subjects())"
```

On a cluster, source (or `module load`) FreeSurfer and FSL **in the job
script**, not only in a login shell. A Python virtualenv does not provide
those binaries.

Package and interpreter details:

```bash
pyfsviz --debug-info
```

## Batch failures

`gen_batch_reports(..., skip_failed=True)` (the default) stores exceptions in
the returned dict instead of aborting:

```python
from pathlib import Path

for subject, result in results.items():
    if not isinstance(result, Path):
        print(subject, result)
```

Set `skip_failed=False` when you want the first error to raise.

## Wrong subjects tree

`SetUpFreeSurfer.sh` often sets `SUBJECTS_DIR` to FreeSurfer’s bundled
`subjects` folder. Point it at **your** recon-all output, or pass
`subjects_dir=` to `FreeSurfer()`.

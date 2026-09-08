"""Generate pyFSViz HTML reports for OpenNeuro ds004731.

Download the dataset separately, run FreeSurfer reconstructions into
``derivatives/freesurfer/``, then source FreeSurfer and FSL in this shell
before running the script. See the user-guide Prerequisites page.

Dataset: https://doi.org/10.18112/openneuro.ds004731.v1.0.0

Example::

    export FREESURFER_HOME=/path/to/freesurfer
    source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
    # also source FSL so flirt is on PATH
    python generate_reports.py /path/to/ds004731-1.0.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pyfsviz import FreeSurfer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to the OpenNeuro ds004731 dataset root "
        "(contains participants.tsv and derivatives/freesurfer/)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Report output directory "
        "(default: <dataset_root>/derivatives/fsqc)",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    subjects_dir = dataset_root / "derivatives" / "freesurfer"
    output_dir = (args.output_dir or dataset_root / "derivatives" / "fsqc").expanduser().resolve()
    participants_tsv = dataset_root / "participants.tsv"

    if not subjects_dir.is_dir():
        raise SystemExit(f"FreeSurfer subjects directory not found: {subjects_dir}")
    if not participants_tsv.is_file():
        raise SystemExit(f"participants.tsv not found: {participants_tsv}")

    fs = FreeSurfer(subjects_dir=str(subjects_dir))

    print(f">> Generating individual reports under {output_dir}")
    results = fs.gen_batch_reports(str(output_dir))
    n_ok = sum(1 for value in results.values() if not isinstance(value, Exception))
    print(f"   {n_ok}/{len(results)} subject report(s) written")

    participants = pd.read_csv(participants_tsv, sep="\t")
    females = participants.loc[participants["sex"] == "F", "participant_id"].tolist()
    males = participants.loc[participants["sex"] == "M", "participant_id"].tolist()

    print(f">> Generating group report ({len(females)} female, {len(males)} male)")
    group_html = fs.gen_group_report(
        str(output_dir),
        groups={"females": females, "males": males},
    )
    print(f"   {group_html}")


if __name__ == "__main__":
    main()

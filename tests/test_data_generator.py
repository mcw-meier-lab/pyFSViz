"""Test data generator for FreeSurfer testing."""

import struct
import tempfile
from collections.abc import Generator
from pathlib import Path

import nibabel as nib
import numpy as np


def create_synthetic_brain_mask(shape: tuple[int, int, int]) -> np.ndarray:
    """Create a synthetic brain-like mask."""
    mask = np.zeros(shape, dtype=np.float32)

    # Create ellipsoid brain shape
    center_x, center_y, center_z = shape[0] // 2, shape[1] // 2, shape[2] // 2
    radius_x, radius_y, radius_z = shape[0] // 3, shape[1] // 3, shape[2] // 3

    x, y, z = np.ogrid[: shape[0], : shape[1], : shape[2]]

    # Ellipsoid equation
    brain_mask = ((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2 + (
        (z - center_z) / radius_z
    ) ** 2 <= 1

    mask[brain_mask] = 1.0

    # Add some noise for realism
    noise = np.random.normal(0, 0.05, shape)
    return np.clip(mask + noise, 0, 1)


def create_minimal_mgz_files(mri_dir: Path, shape: tuple[int, int, int] = (256, 256, 256)) -> None:
    """Create minimal .mgz files for testing."""
    brain_mask = create_synthetic_brain_mask(shape)

    # Create affine transformation matrix (identity for simplicity)
    affine = np.eye(4)

    # orig.mgz - original T1
    orig_data = np.random.rand(*shape).astype(np.float32) * brain_mask
    orig_img = nib.Nifti1Image(orig_data, affine)
    nib.save(orig_img, mri_dir / "orig.mgz")

    # T1.mgz - normalized T1
    t1_data = orig_data * 0.8 + np.random.normal(0, 0.1, shape) * brain_mask
    t1_img = nib.Nifti1Image(t1_data, affine)
    nib.save(t1_img, mri_dir / "T1.mgz")

    # brainmask.mgz - brain mask
    brain_img = nib.Nifti1Image(brain_mask, affine)
    nib.save(brain_img, mri_dir / "brainmask.mgz")

    # wm.mgz - white matter segmentation
    wm_data = np.zeros(shape, dtype=np.float32)
    # Create white matter regions (simplified)
    wm_data[shape[0] // 3 : 2 * shape[0] // 3, shape[1] // 3 : 2 * shape[1] // 3, shape[2] // 3 : 2 * shape[2] // 3] = 1
    wm_data = wm_data * brain_mask
    wm_img = nib.Nifti1Image(wm_data, affine)
    nib.save(wm_img, mri_dir / "wm.mgz")

    # aparc+aseg.mgz - parcellation with realistic labels
    aparc_data = np.zeros(shape, dtype=np.int32)
    # Add some realistic FreeSurfer labels
    labels = [
        2,
        3,
        4,
        5,
        7,
        8,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        24,
        26,
        28,
        30,
        31,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        58,
        60,
        62,
        63,
        77,
        78,
        79,
        80,
        81,
        82,
        85,
        251,
        252,
        253,
        254,
        255,
    ]

    # Randomly assign labels to brain voxels
    brain_voxels = np.where(brain_mask > 0.5)
    for i in range(len(brain_voxels[0])):
        if np.random.random() < 0.3:  # Only label 30% of voxels
            aparc_data[brain_voxels[0][i], brain_voxels[1][i], brain_voxels[2][i]] = np.random.choice(labels)

    aparc_img = nib.Nifti1Image(aparc_data, affine)
    nib.save(aparc_img, mri_dir / "aparc+aseg.mgz")

    # ribbon files for left and right hemispheres
    for hemi in ["lh", "rh"]:
        ribbon_data = np.zeros(shape, dtype=np.float32)
        # Create hemisphere-specific ribbon
        if hemi == "lh":
            ribbon_data[:, : shape[1] // 2, :] = brain_mask[:, : shape[1] // 2, :] * 0.8
        else:
            ribbon_data[:, shape[1] // 2 :, :] = brain_mask[:, shape[1] // 2 :, :] * 0.8

        ribbon_img = nib.Nifti1Image(ribbon_data, affine)
        nib.save(ribbon_img, mri_dir / f"{hemi}.ribbon.mgz")


def create_minimal_transform_files(transforms_dir: Path) -> None:
    """Create minimal transform files."""
    # talairach.xfm.lta - FreeSurfer LTA format
    with open(transforms_dir / "talairach.xfm.lta", "w") as f:
        f.write("# LTA file\n")
        f.write("# Transform matrix\n")
        f.write("# Created for testing\n")
        f.write("# FreeSurfer format\n")
        f.write("# Matrix data follows\n")
        f.write("1.00000000 0.00000000 0.00000000 0.00000000\n")
        f.write("0.00000000 1.00000000 0.00000000 0.00000000\n")
        f.write("0.00000000 0.00000000 1.00000000 0.00000000\n")
        f.write("0.00000000 0.00000000 0.00000000 1.00000000\n")

    # talairach.lta (copy for compatibility)
    with open(transforms_dir / "talairach.lta", "w") as f:
        f.write("# LTA file\n")
        f.write("# Transform matrix\n")
        f.write("# Created for testing\n")
        f.write("# FreeSurfer format\n")
        f.write("# Matrix data follows\n")
        f.write("1.00000000 0.00000000 0.00000000 0.00000000\n")
        f.write("0.00000000 1.00000000 0.00000000 0.00000000\n")
        f.write("0.00000000 0.00000000 1.00000000 0.00000000\n")
        f.write("0.00000000 0.00000000 0.00000000 1.00000000\n")


def create_minimal_surface_files(surf_dir: Path) -> None:
    """Create minimal surface files with valid geometry."""
    for hemi in ["lh", "rh"]:
        for surf_type in ["pial", "inflated", "white", "sulc"]:
            surface_file = surf_dir / f"{hemi}.{surf_type}"

            if surf_type == "sulc":
                # Sulc files are curvature files, not surface files
                create_sulc_file(surface_file)
            else:
                # Create proper FreeSurfer surface file
                create_surface_file(surface_file)


def create_surface_file(surface_file: Path) -> None:
    """Create a minimal but valid FreeSurfer surface file."""
    # Create a simple tetrahedron (4 vertices, 4 faces)
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # vertex 0
            [1.0, 0.0, 0.0],  # vertex 1
            [0.5, 1.0, 0.0],  # vertex 2
            [0.5, 0.5, 1.0],  # vertex 3
        ],
        dtype=np.float32,
    )

    faces = np.array(
        [
            [0, 1, 2],  # face 0
            [0, 1, 3],  # face 1
            [1, 2, 3],  # face 2
            [0, 2, 3],  # face 3
        ],
        dtype=np.int32,
    )

    # Write FreeSurfer surface file format
    with open(surface_file, "wb") as f:
        # Magic number
        f.write(b"\xff\xff\xfe")

        # Write vertices
        f.write(struct.pack(">I", vertices.shape[0]))  # number of vertices
        for vertex in vertices:
            f.write(struct.pack(">fff", vertex[0], vertex[1], vertex[2]))

        # Write faces
        f.write(struct.pack(">I", faces.shape[0]))  # number of faces
        for face in faces:
            f.write(struct.pack(">III", face[0], face[1], face[2]))


def create_sulc_file(sulc_file: Path) -> None:
    """Create a minimal sulc (curvature) file."""
    # Create simple curvature values for 4 vertices
    curvatures = np.array([0.1, -0.2, 0.3, -0.1], dtype=np.float32)

    # Write FreeSurfer curvature file format
    with open(sulc_file, "wb") as f:
        # Magic number
        f.write(b"\xff\xff\xfe")

        # Write number of vertices
        f.write(struct.pack(">I", len(curvatures)))

        # Write curvature values
        for curv in curvatures:
            f.write(struct.pack(">f", curv))


def create_minimal_annotation_files(label_dir: Path) -> None:
    """Create minimal annotation files."""
    for hemi in ["lh", "rh"]:
        # Create minimal annotation file
        annot_file = label_dir / f"{hemi}.aparc.annot"
        create_annotation_file(annot_file)


def create_annotation_file(annot_file: Path) -> None:
    """Create a minimal but valid FreeSurfer annotation file."""
    # Number of vertices (should match our surface files)
    num_vertices = 4

    # Create labels for each vertex
    labels = np.array([0, 1, 2, 3], dtype=np.int32)  # Simple labels

    # Create colors for each label (RGBA format)
    colors = np.array(
        [
            [0, 0, 0, 0],  # label 0: transparent
            [255, 0, 0, 0],  # label 1: red
            [0, 255, 0, 0],  # label 2: green
            [0, 0, 255, 0],  # label 3: blue
        ],
        dtype=np.int32,
    )

    # Write FreeSurfer annotation file format
    with open(annot_file, "wb") as f:
        # Write number of vertices
        f.write(struct.pack(">I", num_vertices))

        # Write labels for each vertex
        for label in labels:
            f.write(struct.pack(">I", label))

        # Write number of color entries
        f.write(struct.pack(">I", len(colors)))

        # Write color table
        for color in colors:
            f.write(struct.pack(">IIII", color[0], color[1], color[2], color[3]))


def create_minimal_recon_log(scripts_dir: Path) -> None:
    """Create minimal recon-all.log file."""
    with open(scripts_dir / "recon-all.log", "w") as f:
        f.write("FreeSurfer recon-all started\n")
        f.write("Processing steps...\n")
        f.write("recon-all finished without error\n")


def create_mock_freesurfer_subject(
    subject_id: str,
    output_dir: Path,
    shape: tuple[int, int, int] = (256, 256, 256),
) -> Path:
    """Create a complete mock FreeSurfer subject directory structure."""
    subject_dir = output_dir / subject_id

    # Create directory structure
    mri_dir = subject_dir / "mri"
    surf_dir = subject_dir / "surf"
    label_dir = subject_dir / "label"
    scripts_dir = subject_dir / "scripts"
    transforms_dir = mri_dir / "transforms"

    for dir_path in [mri_dir, surf_dir, label_dir, scripts_dir, transforms_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Create all required files
    create_minimal_mgz_files(mri_dir, shape)
    create_minimal_transform_files(transforms_dir)
    create_minimal_surface_files(surf_dir)
    create_minimal_annotation_files(label_dir)
    create_minimal_recon_log(scripts_dir)

    return subject_dir


def create_mock_freesurfer_home(output_dir: Path) -> Path:
    """Create a mock FreeSurfer home directory."""
    freesurfer_home = output_dir / "freesurfer"
    freesurfer_home.mkdir(parents=True, exist_ok=True)

    # Create FreeSurferColorLUT.txt
    lut_file = freesurfer_home / "FreeSurferColorLUT.txt"
    with open(lut_file, "w") as f:
        f.write("# FreeSurfer Color Look Up Table\n")
        f.write("# No. Label Name: R G B A\n")
        f.write("0   Unknown 0 0 0 0\n")
        f.write("1   Left-Cerebral-Exterior 70 130 180 0\n")
        f.write("2   Left-Cerebral-White-Matter 245 245 245 0\n")
        f.write("3   Left-Cerebral-Cortex 205 62 78 0\n")
        f.write("4   Left-Lateral-Ventricle 120 18 134 0\n")
        f.write("5   Left-Inf-Lat-Vent 196 58 250 0\n")
        f.write("7   Left-Cerebellum-Exterior 0 148 0 0\n")
        f.write("8   Left-Cerebellum-White-Matter 220 248 164 0\n")
        f.write("10  Left-Thalamus-Proper 230 148 34 0\n")
        f.write("11  Left-Caudate 60 58 210 0\n")
        f.write("12  Left-Putamen 60 58 210 0\n")
        f.write("13  Left-Pallidum 25 155 0 0\n")
        f.write("14  Left-Hippocampus 25 155 0 0\n")
        f.write("15  Left-Amygdala 25 155 0 0\n")
        f.write("16  Left-Accumbens-area 25 155 0 0\n")
        f.write("17  Left-Substancia-Nigra 25 155 0 0\n")
        f.write("18  Left-VentralDC 25 155 0 0\n")
        f.write("24  CSF 120 18 134 0\n")
        f.write("26  Left-vessel 0 118 14 0\n")
        f.write("28  Left-choroid-plexus 122 135 50 0\n")
        f.write("30  Left-F3orb 250 250 0 0\n")
        f.write("31  Left-lOg 255 165 0 0\n")
        f.write("41  Right-Cerebral-Exterior 70 130 180 0\n")
        f.write("42  Right-Cerebral-White-Matter 245 245 245 0\n")
        f.write("43  Right-Cerebral-Cortex 205 62 78 0\n")
        f.write("44  Right-Lateral-Ventricle 120 18 134 0\n")
        f.write("45  Right-Inf-Lat-Vent 196 58 250 0\n")
        f.write("46  Right-Cerebellum-Exterior 0 148 0 0\n")
        f.write("47  Right-Cerebellum-White-Matter 220 248 164 0\n")
        f.write("48  Right-Thalamus-Proper 230 148 34 0\n")
        f.write("49  Right-Caudate 60 58 210 0\n")
        f.write("50  Right-Putamen 60 58 210 0\n")
        f.write("51  Right-Pallidum 25 155 0 0\n")
        f.write("52  Right-Hippocampus 25 155 0 0\n")
        f.write("53  Right-Amygdala 25 155 0 0\n")
        f.write("54  Right-Accumbens-area 25 155 0 0\n")
        f.write("58  Right-Substancia-Nigra 25 155 0 0\n")
        f.write("60  Right-VentralDC 25 155 0 0\n")
        f.write("62  Right-vessel 0 118 14 0\n")
        f.write("63  Right-choroid-plexus 122 135 50 0\n")
        f.write("77  WM-hypointensities 25 155 0 0\n")
        f.write("78  Left-WM-hypointensities 25 155 0 0\n")
        f.write("79  Right-WM-hypointensities 25 155 0 0\n")
        f.write("80  Non-WM-hypointensities 25 155 0 0\n")
        f.write("81  Left-non-WM-hypointensities 25 155 0 0\n")
        f.write("82  Right-non-WM-hypointensities 25 155 0 0\n")
        f.write("85  Optic-Chiasm 25 155 0 0\n")
        f.write("251 Left-unknown 25 155 0 0\n")
        f.write("252 Right-unknown 25 155 0 0\n")
        f.write("253 Left-undefined 25 155 0 0\n")
        f.write("254 Right-undefined 25 155 0 0\n")
        f.write("255 Left-undefined 25 155 0 0\n")

    return freesurfer_home


def setup_mock_freesurfer_environment() -> Generator[tuple[Path, Path], None, None]:
    """Set up a complete mock FreeSurfer environment for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create FreeSurfer home
        freesurfer_home = create_mock_freesurfer_home(tmp_path)

        # Create subjects directory
        subjects_dir = tmp_path / "subjects"
        subjects_dir.mkdir(parents=True, exist_ok=True)

        # Create test subject
        create_mock_freesurfer_subject("sub-001", subjects_dir)

        yield freesurfer_home, subjects_dir

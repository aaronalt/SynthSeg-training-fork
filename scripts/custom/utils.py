#!/usr/bin/env python3
"""
Utility functions for working with SynthMorph preprocessing outputs
and warping claustrum segmentations between spaces.
"""

import os
import subprocess
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List


class SynthMorphFiles:
    """
    Handles finding and organizing SynthMorph preprocessing files for a subject.
    """

    def __init__(self, subject_dir: str):
        """
        Initialize with path to subject's anat directory.

        Parameters
        ----------
        subject_dir : str
            Path to subject's anat directory, e.g.:
            /path/to/sub-6177/ses-T5/anat
        """
        self.subject_dir = Path(subject_dir)
        self.synthmorph_dir = self.subject_dir / 'synthmorph'

        # Extract subject/session info
        parts = str(self.subject_dir).split('/')
        self.subject = next((p for p in parts if p.startswith('sub-')), None)
        self.session = next((p for p in parts if p.startswith('ses-')), None)

        if not self.synthmorph_dir.exists():
            raise FileNotFoundError(f"SynthMorph directory not found: {self.synthmorph_dir}")

        # Find all files
        self._find_files()

    def _find_files(self):
        """Locate all relevant files in the synthmorph directory."""

        # Required files
        self.files = {
            # Cropped hemispheres (most important!)
            'lh_crop': self.subject_dir / 'invol.lh.crop.nii.gz',
            'rh_crop': self.subject_dir / 'invol.rh.crop.nii.gz',

            # Warps
            'warp_to_mni': self.synthmorph_dir / 'warp.to.mni152.1.5mm.1.0mm.nii.gz',
            'warp_to_native': self.synthmorph_dir / 'warp.to.mni152.1.5mm.1.0mm.inv.nii.gz',

            # Transforms
            'aff_lta': self.synthmorph_dir / 'aff.lta',
            'reg_invol_to_targ': self.synthmorph_dir / 'reg.invol_to_targ.lta',
            'reg_targ_to_invol': self.synthmorph_dir / 'reg.targ_to_invol.lta',

            # Reference images
            'invol_crop': self.synthmorph_dir / 'invol.crop.nii.gz',
            'invol_stripped': self.synthmorph_dir / 'invol.stripped.mgz',
            'morph_out': self.synthmorph_dir / 'morph.out.nii.gz',

            # Original input (may be in parent dir)
            'original_t1w': None  # Will search for this
        }

        # Find original T1w
        possible_t1w = list(self.subject_dir.glob('*_T1w.nii.gz'))
        if possible_t1w:
            self.files['original_t1w'] = possible_t1w[0]

        # Validate critical files exist
        critical_files = ['lh_crop', 'rh_crop', 'warp_to_mni', 'warp_to_native']
        missing = [k for k in critical_files if not self.files[k].exists()]

        if missing:
            raise FileNotFoundError(
                f"Missing critical files: {missing}\n"
                f"Searched in: {self.subject_dir}"
            )

    def get(self, file_key: str) -> Path:
        """Get path to a specific file."""
        if file_key not in self.files:
            raise KeyError(f"Unknown file key: {file_key}")

        file_path = self.files[file_key]
        if file_path is None or not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_key}")

        return file_path

    def info(self) -> Dict:
        """Return dictionary with subject info and file paths."""
        return {
            'subject': self.subject,
            'session': self.session,
            'subject_dir': str(self.subject_dir),
            'files': {k: str(v) if v else None for k, v in self.files.items()},
            'files_exist': {k: v.exists() if v else False for k, v in self.files.items()}
        }

    @staticmethod
    def find_all_subjects(base_dir: str, pattern: str = 'sub-*/ses-*/anat') -> List['SynthMorphFiles']:
        """
        Find all subjects with SynthMorph preprocessing.

        Parameters
        ----------
        base_dir : str
            Base directory to search
        pattern : str
            Glob pattern for finding subject directories

        Returns
        -------
        list of SynthMorphFiles
            List of initialized objects for each found subject
        """
        from glob import glob

        search_pattern = os.path.join(base_dir, pattern)
        subject_dirs = []

        for path in glob(search_pattern):
            # Check if this directory has the required files
            try:
                sm_files = SynthMorphFiles(path)
                subject_dirs.append(sm_files)
            except (FileNotFoundError, Exception):
                continue

        return subject_dirs


class ClaustrumWarper:
    """
    Handles warping of images and segmentations between native and MNI space
    using SynthMorph preprocessing outputs.
    """

    def __init__(self, synthmorph_files: SynthMorphFiles, mni_template: Optional[str] = None):
        """
        Initialize warper.

        Parameters
        ----------
        synthmorph_files : SynthMorphFiles
            Object containing paths to preprocessing files
        mni_template : str, optional
            Path to MNI152 template. If None, uses FreeSurfer's default.
        """
        self.sm_files = synthmorph_files

        # Set MNI template
        if mni_template is None:
            freesurfer_home = os.environ.get('FREESURFER_HOME', '/home/aaron/freesurfer')
            self.mni_template = os.path.join(
                freesurfer_home,
                'average/mni_icbm152_nlin_asym_09c/mni152.1.0mm.nii.gz'
            )
        else:
            self.mni_template = mni_template

        if not os.path.exists(self.mni_template):
            raise FileNotFoundError(f"MNI template not found: {self.mni_template}")

    def warp_to_mni(self, input_img: str, output_img: str,
                    interpolation: str = 'trilinear') -> str:
        """
        Warp an image from native space to MNI space.

        Parameters
        ----------
        input_img : str
            Path to input image in native space
        output_img : str
            Path for output image in MNI space
        interpolation : str
            Interpolation method: 'trilinear', 'nearest', 'cubic'

        Returns
        -------
        str
            Path to output image
        """

        warp_file = str(self.sm_files.get('warp_to_mni'))

        # Use mri_convert with warp field
        cmd = [
            'mri_convert',
            '-rt', interpolation,
            '-at', warp_file,
            input_img,
            output_img
        ]

        print(f"Warping to MNI: {os.path.basename(input_img)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"mri_convert failed:\n{result.stderr}")

        if not os.path.exists(output_img):
            raise RuntimeError(f"Output not created: {output_img}")

        print(f"  → {os.path.basename(output_img)}")

        return output_img

    def warp_to_native(self, input_img: str, output_img: str,
                       interpolation: str = 'trilinear',
                       reference: Optional[str] = None) -> str:
        """
        Warp an image from MNI space to native space.

        Parameters
        ----------
        input_img : str
            Path to input image in MNI space
        output_img : str
            Path for output image in native space
        interpolation : str
            Interpolation method: 'trilinear', 'nearest', 'cubic'
            Use 'nearest' for label images!
        reference : str, optional
            Reference image defining native space geometry.
            If None, uses invol.stripped.mgz

        Returns
        -------
        str
            Path to output image
        """

        warp_file = str(self.sm_files.get('warp_to_native'))

        if reference is None:
            reference = str(self.sm_files.get('invol_stripped'))

        # Use mri_convert with inverse warp
        cmd = [
            'mri_convert',
            '-rt', interpolation,
            '-at', warp_file,
            '-rl', reference,  # Reference "like" - defines output geometry
            input_img,
            output_img
        ]

        print(f"Warping to native: {os.path.basename(input_img)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"mri_convert failed:\n{result.stderr}")

        if not os.path.exists(output_img):
            raise RuntimeError(f"Output not created: {output_img}")

        print(f"  → {os.path.basename(output_img)}")

        return output_img

    def create_mni_crops(self, output_dir: str, crop_size_mm: float = 60) -> Tuple[str, str]:
        """
        Create cropped regions around claustrum in MNI space.

        This warps the native hemisphere crops to MNI, then creates
        crops around the claustrum region for both hemispheres.

        Parameters
        ----------
        output_dir : str
            Directory to save outputs
        crop_size_mm : float
            Size of crop in mm (default: 60mm)

        Returns
        -------
        tuple
            (lh_mni_crop_path, rh_mni_crop_path)
        """
        os.makedirs(output_dir, exist_ok=True)

        # Warp hemisphere crops to MNI
        lh_native = str(self.sm_files.get('lh_crop'))
        rh_native = str(self.sm_files.get('rh_crop'))

        lh_mni_full = os.path.join(output_dir, 'lh_warped_to_mni.nii.gz')
        rh_mni_full = os.path.join(output_dir, 'rh_warped_to_mni.nii.gz')

        self.warp_to_mni(lh_native, lh_mni_full)
        self.warp_to_mni(rh_native, rh_mni_full)

        # Load MNI template to get standard space dimensions
        mni = nib.load(self.mni_template)
        mni_shape = mni.shape
        mni_affine = mni.affine

        # Define expected claustrum center in MNI space (approximate)
        # These are rough coordinates - adjust based on your atlas
        lh_center_mni = [60, 128, 96]  # LH claustrum in MNI voxels (1mm)
        rh_center_mni = [135, 128, 96]  # RH claustrum in MNI voxels (1mm)

        # Create crops
        lh_crop = self._crop_around_center(
            lh_mni_full, lh_center_mni, crop_size_mm,
            os.path.join(output_dir, 'lh_mni_crop.nii.gz')
        )

        rh_crop = self._crop_around_center(
            rh_mni_full, rh_center_mni, crop_size_mm,
            os.path.join(output_dir, 'rh_mni_crop.nii.gz')
        )

        return lh_crop, rh_crop

    def _crop_around_center(self, img_path: str, center_vox: List[int],
                            crop_size_mm: float, output_path: str) -> str:
        """Helper to crop around a center point."""

        img = nib.load(img_path)
        data = img.get_fdata()
        affine = img.affine

        # Calculate crop size in voxels
        voxel_size = np.abs(np.diag(affine)[:3])
        crop_vox = [int(crop_size_mm / vs) for vs in voxel_size]

        # Define crop bounds
        slices = []
        for i in range(3):
            start = max(0, center_vox[i] - crop_vox[i] // 2)
            end = min(data.shape[i], center_vox[i] + crop_vox[i] // 2)
            slices.append(slice(start, end))

        # Crop data
        cropped_data = data[slices[0], slices[1], slices[2]]

        # Update affine
        new_origin = np.dot(affine, [slices[i].start for i in range(3)] + [1])[:3]
        cropped_affine = affine.copy()
        cropped_affine[:3, 3] = new_origin

        # Save
        cropped_img = nib.Nifti1Image(cropped_data, cropped_affine, img.header)
        nib.save(cropped_img, output_path)

        return output_path


def check_image_properties(img_path: str) -> Dict:
    """
    Check properties of an image file.

    Parameters
    ----------
    img_path : str
        Path to image

    Returns
    -------
    dict
        Dictionary with image properties
    """
    img = nib.load(img_path)
    data = img.get_fdata()

    props = {
        'path': img_path,
        'shape': data.shape,
        'dtype': data.dtype,
        'voxel_size': np.abs(np.diag(img.affine)[:3]),
        'min': float(data.min()),
        'max': float(data.max()),
        'mean': float(data.mean()),
        'std': float(data.std()),
        'non_zero_voxels': int(np.count_nonzero(data)),
        'unique_values': len(np.unique(data)),
        'is_empty': np.allclose(data, 0),
    }

    # Check if it's a label image
    unique_vals = np.unique(data)
    if len(unique_vals) < 20 and np.all(unique_vals == unique_vals.astype(int)):
        props['appears_to_be_labels'] = True
        props['unique_labels'] = unique_vals.tolist()
    else:
        props['appears_to_be_labels'] = False

    return props


def print_image_properties(img_path: str):
    """Pretty print image properties."""
    props = check_image_properties(img_path)

    print(f"\nImage: {os.path.basename(props['path'])}")
    print(f"  Shape: {props['shape']}")
    print(f"  Voxel size: {props['voxel_size']}")
    print(f"  Data range: [{props['min']:.2f}, {props['max']:.2f}]")
    print(f"  Mean ± std: {props['mean']:.2f} ± {props['std']:.2f}")
    print(f"  Non-zero voxels: {props['non_zero_voxels']}")
    print(f"  Unique values: {props['unique_values']}")
    print(f"  Is empty: {props['is_empty']}")

    if props['appears_to_be_labels']:
        print(f"  Labels present: {props['unique_labels']}")
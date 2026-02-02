#!/usr/bin/env python3
"""
Subject-aware stratified train/test split for SynthSeg training.

Splits by SUBJECT ID so all modalities of a subject stay in the same split
(no data leakage). Supports multiple base directories (e.g., 7T and 3T).

Usage:
    python create_train_test_split_stratified.py --method symlink --test_ratio 0.2
"""

import argparse
import shutil
import random
import re
import math
from pathlib import Path
from collections import defaultdict
import json
import numpy as np


def extract_subject_id(filename):
    """Extract subject ID (e.g., 'sub-6351') from a filename like
    sub-6351_ses-T4_t1w_lh_training_label.nii.gz"""
    match = re.match(r'(sub-\d+)', filename)
    return match.group(1) if match else filename


def collect_all_labels(base_dirs_with_subdirs):
    """Collect all label files from multiple base directories and subdirectories.

    Args:
        base_dirs_with_subdirs: list of dicts, each with:
            - 'base_dir': path to base directory
            - 'subdirs': list of subdirectory names
            - 'field_strength': label like '7T' or '3T'

    Returns:
        list of file info dicts
    """
    all_files = []

    for entry in base_dirs_with_subdirs:
        base_path = Path(entry['base_dir'])
        field_strength = entry.get('field_strength', '')

        for subdir in entry['subdirs']:
            subdir_path = base_path / subdir
            if not subdir_path.exists():
                print(f"  Warning: {subdir_path} does not exist, skipping")
                continue

            files = sorted(list(subdir_path.glob('*.nii.gz')) + list(subdir_path.glob('*.nii')))
            print(f"  Found {len(files)} files in {field_strength}/{subdir}/")

            for f in files:
                subject_id = extract_subject_id(f.name)
                all_files.append({
                    'path': f,
                    'name': f.name,
                    'source': subdir,
                    'field_strength': field_strength,
                    'subject_id': subject_id,
                })

    return all_files


def create_subject_split(all_files, test_ratio, seed=42):
    """Split by SUBJECT ID so all files for a subject are in the same set.

    Groups subjects, then assigns entire subjects to train or test.
    Stratifies by field strength to ensure both 7T and 3T subjects
    appear in train and test.
    """
    random.seed(seed)

    # Group files by (field_strength, subject_id)
    subjects_by_fs = defaultdict(lambda: defaultdict(list))
    for f in all_files:
        subjects_by_fs[f['field_strength']][f['subject_id']].append(f)

    train_files = []
    test_files = []
    train_subjects = []
    test_subjects = []

    print(f"\n{'=' * 60}")
    print(f"Subject-Aware Stratified Split (test_ratio: {test_ratio})")
    print(f"{'=' * 60}")

    for fs, subjects_dict in sorted(subjects_by_fs.items()):
        subject_ids = sorted(subjects_dict.keys())
        random.shuffle(subject_ids)

        n_test = max(1, int(len(subject_ids) * test_ratio))

        test_subs = subject_ids[:n_test]
        train_subs = subject_ids[n_test:]

        for sub in test_subs:
            test_files.extend(subjects_dict[sub])
            test_subjects.append((fs, sub))
        for sub in train_subs:
            train_files.extend(subjects_dict[sub])
            train_subjects.append((fs, sub))

        n_train_files = sum(len(subjects_dict[s]) for s in train_subs)
        n_test_files = sum(len(subjects_dict[s]) for s in test_subs)
        print(f"  {fs:>4s}: {len(train_subs)} train subjects ({n_train_files} files), "
              f"{len(test_subs)} test subjects ({n_test_files} files)")

    print(f"\n  Train subjects: {[s for _, s in sorted(train_subjects)]}")
    print(f"  Test subjects:  {[s for _, s in sorted(test_subjects)]}")

    return train_files, test_files


def compute_sampling_probabilities(file_list, mode='sqrt'):
    """Compute per-file sampling probabilities for training.

    Args:
        file_list: list of file info dicts (must have 'field_strength' and 'source')
        mode: weighting strategy
            - 'uniform': equal probability per file (natural ratio)
            - 'balanced': equal probability per field_strength group
            - 'sqrt': square root balancing (compromise between uniform and balanced)
            - 'custom': dict of {field_strength: weight}, e.g. {'3T': 0.5, '7T': 0.5}

    Returns:
        numpy array of probabilities (sums to 1.0), same order as file_list
    """
    if isinstance(mode, dict):
        # Custom weights per field strength
        custom_weights = mode
        mode = 'custom'

    # Group by field strength
    fs_counts = defaultdict(int)
    for f in file_list:
        fs_counts[f['field_strength']] += 1

    n_total = len(file_list)
    probs = np.zeros(n_total)

    if mode == 'uniform':
        probs[:] = 1.0 / n_total

    elif mode == 'balanced':
        # Each field strength gets equal total weight
        n_groups = len(fs_counts)
        for i, f in enumerate(file_list):
            group_size = fs_counts[f['field_strength']]
            probs[i] = (1.0 / n_groups) / group_size

    elif mode == 'sqrt':
        # Weight proportional to sqrt of group size (reduces imbalance without
        # over-sampling small groups as aggressively as 'balanced')
        sqrt_sizes = {fs: math.sqrt(count) for fs, count in fs_counts.items()}
        total_sqrt = sum(sqrt_sizes.values())
        for i, f in enumerate(file_list):
            fs = f['field_strength']
            group_weight = sqrt_sizes[fs] / total_sqrt
            probs[i] = group_weight / fs_counts[fs]

    elif mode == 'custom':
        total_weight = sum(custom_weights.values())
        for i, f in enumerate(file_list):
            fs = f['field_strength']
            w = custom_weights.get(fs, 1.0) / total_weight
            probs[i] = w / fs_counts[fs]

    # Normalize
    probs = probs / probs.sum()
    return probs


def populate_directories(train_files, test_files, train_path, test_path, method):
    """Populate directories with source-tagged filenames to avoid collisions."""

    res_map = {
        't1w': '7T_0.6mm',
        't2w-cor': '7T_0.22mm',
        't2w-tra': '7T_0.22mm',
        '3t-t1w': '3T_1.0mm',
    }

    for folder_type, files, target_path in [('train', train_files, train_path),
                                            ('test', test_files, test_path)]:
        for file_info in files:
            src = file_info['path']
            source_tag = file_info['source']
            fs = file_info['field_strength']

            # Build prefix: use res_map if available, otherwise field_strength_source
            prefix = res_map.get(source_tag, f"{fs}_{source_tag}")

            new_name = f"{prefix}_{file_info['name']}"
            dst = target_path / new_name

            if method == 'symlink':
                dst.symlink_to(src.resolve())
            elif method == 'copy':
                shutil.copy2(src, dst)

    print(f"  Processed {len(train_files)} train + {len(test_files)} test files")


def save_split_info(output_dir, train_files, test_files, config):
    """Save split information for reproducibility"""
    output_path = Path(output_dir)

    split_info = {
        'config': config,
        'n_train': len(train_files),
        'n_test': len(test_files),
        'train_subjects': sorted(set(f['subject_id'] for f in train_files)),
        'test_subjects': sorted(set(f['subject_id'] for f in test_files)),
        'train_files': [
            {
                'name': f['name'],
                'source': f['source'],
                'field_strength': f['field_strength'],
                'subject_id': f['subject_id'],
                'path': str(f['path'])
            } for f in train_files
        ],
        'test_files': [
            {
                'name': f['name'],
                'source': f['source'],
                'field_strength': f['field_strength'],
                'subject_id': f['subject_id'],
                'path': str(f['path'])
            } for f in test_files
        ]
    }

    info_file = output_path / 'split_info.json'
    with open(info_file, 'w') as fp:
        json.dump(split_info, fp, indent=2)
    print(f"  Saved split info to {info_file}")


def print_summary(train_files, test_files):
    """Print summary statistics"""

    print(f"\n{'='*60}")
    print("SPLIT SUMMARY")
    print(f"{'='*60}")

    for label, files in [("Train", train_files), ("Test", test_files)]:
        by_fs_source = defaultdict(int)
        subjects = set()
        for f in files:
            by_fs_source[(f['field_strength'], f['source'])] += 1
            subjects.add(f['subject_id'])

        print(f"\n  {label}: {len(files)} files, {len(subjects)} subjects")
        for (fs, src), count in sorted(by_fs_source.items()):
            print(f"    {fs:>4s}/{src:<12s}: {count:3d} files")

    print(f"\n{'='*60}\n")


def create_train_test_split(base_dirs_with_subdirs,
                           output_dir='/tmp/training_split',
                           method='copy',
                           test_ratio=0.2,
                           seed=42,
                           prob_mode='sqrt',
                           verbose=True):
    """
    Create subject-aware stratified train/test split.

    Args:
        base_dirs_with_subdirs: list of dicts with 'base_dir', 'subdirs', 'field_strength'
        output_dir: Output directory for train/test split
        method: 'symlink' or 'copy'
        test_ratio: Ratio of subjects for testing (0.0-1.0)
        seed: Random seed for reproducibility
        prob_mode: Probability weighting mode ('uniform', 'balanced', 'sqrt', or dict)
        verbose: Print progress messages

    Returns:
        tuple: (train_path, test_path, train_probs, val_probs)
    """

    if verbose:
        print(f"\n{'='*60}")
        print("Subject-Aware Stratified Split for SynthSeg Training")
        print(f"{'='*60}\n")
        print("Collecting files...")

    all_files = collect_all_labels(base_dirs_with_subdirs)

    if len(all_files) == 0:
        if verbose:
            print("\nNo files found!")
        return None, None, None, None

    if verbose:
        print(f"\nTotal files collected: {len(all_files)}")

    # Split by subject
    train_files, test_files = create_subject_split(all_files, test_ratio, seed)

    # Create directories
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)

    train_path = output_path / 'train'
    test_path = output_path / 'test'
    train_path.mkdir(parents=True)
    test_path.mkdir(parents=True)

    # Populate
    populate_directories(train_files, test_files, train_path, test_path, method)

    # Compute sampling probabilities
    train_probs = compute_sampling_probabilities(train_files, mode=prob_mode)
    val_probs = compute_sampling_probabilities(test_files, mode=prob_mode)

    if verbose:
        print(f"\nSampling probabilities (mode: {prob_mode}):")
        fs_weights = defaultdict(float)
        for f, p in zip(train_files, train_probs):
            fs_weights[f['field_strength']] += p
        for fs, w in sorted(fs_weights.items()):
            print(f"  {fs}: {w:.1%} total weight")

    # Save info
    config = {
        'method': method,
        'test_ratio': test_ratio,
        'seed': seed,
        'prob_mode': str(prob_mode),
        'base_dirs': [
            {'base_dir': str(e['base_dir']), 'subdirs': e['subdirs'],
             'field_strength': e['field_strength']}
            for e in base_dirs_with_subdirs
        ]
    }
    save_split_info(output_dir, train_files, test_files, config)

    if verbose:
        print_summary(train_files, test_files)

    return train_path, test_path, train_probs, val_probs


def extract_test_from_validation(val_dir, output_test_dir, test_ratio=0.5, seed=42):
    """Copy a portion of validation set to test directory (subject-aware)."""
    random.seed(seed)
    val_dir = Path(val_dir)
    output_test_dir = Path(output_test_dir)
    output_test_dir.mkdir(parents=True, exist_ok=True)

    all_val_files = sorted(list(val_dir.glob('*nii.gz')))

    # Group by subject
    by_subject = defaultdict(list)
    for f in all_val_files:
        sub = extract_subject_id(f.name)
        by_subject[sub].append(f)

    subjects = sorted(by_subject.keys())
    random.shuffle(subjects)
    n_test = max(1, int(len(subjects) * test_ratio))
    test_subjects = subjects[:n_test]

    count = 0
    for sub in test_subjects:
        for f in by_subject[sub]:
            shutil.copy(f, output_test_dir / f.name)
            count += 1

    print(f"  Copied {count} files ({len(test_subjects)} subjects) to test set")


def main():
    parser = argparse.ArgumentParser(
        description='Subject-aware stratified train/test split for SynthSeg'
    )

    parser.add_argument('--base_dir', type=str, default='/training_labels')
    parser.add_argument('--subdirs', type=str, nargs='+',
                       default=['t1w', 't2w-cor', 't2w-tra'])
    parser.add_argument('--field_strength', type=str, default='7T')
    parser.add_argument('--base_dir_2', type=str, default=None,
                       help='Second base dir (e.g., for 3T data)')
    parser.add_argument('--subdirs_2', type=str, nargs='+', default=['t1w'])
    parser.add_argument('--field_strength_2', type=str, default='3T')
    parser.add_argument('--output_dir', type=str, default='/tmp/training_split')
    parser.add_argument('--method', type=str, choices=['symlink', 'copy'], default='copy')
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prob_mode', type=str, default='sqrt',
                       choices=['uniform', 'balanced', 'sqrt'],
                       help='Sampling probability mode')

    args = parser.parse_args()

    base_dirs = [
        {'base_dir': args.base_dir, 'subdirs': args.subdirs,
         'field_strength': args.field_strength}
    ]
    if args.base_dir_2:
        base_dirs.append(
            {'base_dir': args.base_dir_2, 'subdirs': args.subdirs_2,
             'field_strength': args.field_strength_2}
        )

    create_train_test_split(
        base_dirs_with_subdirs=base_dirs,
        output_dir=args.output_dir,
        method=args.method,
        test_ratio=args.test_ratio,
        seed=args.seed,
        prob_mode=args.prob_mode,
        verbose=True
    )


if __name__ == '__main__':
    main()

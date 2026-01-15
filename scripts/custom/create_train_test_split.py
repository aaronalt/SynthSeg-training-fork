#!/usr/bin/env python3
"""
Combine training labels from multiple directories and create train/test split
for SynthSeg training.

Usage:
    python create_train_test_split.py --method symlink --test_ratio 0.2
    python create_train_test_split.py --method copy --test_ratio 0.15
"""

import argparse
import shutil
import random
from pathlib import Path
import json


def collect_all_labels(base_dir, subdirs):
    """Collect all label files from multiple subdirectories"""
    
    base_path = Path(base_dir)
    all_files = []
    
    for subdir in subdirs:
        subdir_path = base_path / subdir
        if not subdir_path.exists():
            print(f"⚠️  Warning: {subdir_path} does not exist, skipping")
            continue
        
        # Get all nifti files
        files = list(subdir_path.glob('*.nii.gz')) + list(subdir_path.glob('*.nii'))
        print(f"Found {len(files)} files in {subdir}/")
        
        for f in files:
            all_files.append({
                'path': f,
                'name': f.name,
                'source': subdir
            })
    
    return all_files


def create_split(all_files, test_ratio, seed=42):
    """Create train/test split"""
    
    # Set seed for reproducibility
    random.seed(seed)
    
    # Shuffle files
    files_copy = all_files.copy()
    random.shuffle(files_copy)
    
    # Calculate split
    n_test = int(len(files_copy) * test_ratio)
    n_train = len(files_copy) - n_test
    
    test_files = files_copy[:n_test]
    train_files = files_copy[n_test:]
    
    print(f"\n{'='*60}")
    print(f"Split created: {n_train} train, {n_test} test")
    print(f"{'='*60}")
    
    return train_files, test_files


def create_directories(output_dir, method):
    """Create output directory structure"""
    
    output_path = Path(output_dir)
    train_path = output_path / 'train'
    test_path = output_path / 'test'
    
    # Remove existing directories if they exist
    if output_path.exists():
        print(f"\n⚠️  Output directory {output_dir} already exists")
        response = input("Remove and recreate? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return None, None
        shutil.rmtree(output_path)
    
    # Create fresh directories
    train_path.mkdir(parents=True)
    test_path.mkdir(parents=True)
    
    print(f"✓ Created {train_path}")
    print(f"✓ Created {test_path}")
    
    return train_path, test_path


def populate_directories(train_files, test_files, train_path, test_path, method):
    """Populate train/test directories with files"""
    
    print(f"\nPopulating directories using method: {method}")
    
    # Process train files
    for file_info in train_files:
        src = file_info['path']
        dst = train_path / file_info['name']
        
        if method == 'symlink':
            dst.symlink_to(src.resolve())
        elif method == 'copy':
            shutil.copy2(src, dst)
    
    print(f"✓ Processed {len(train_files)} train files")
    
    # Process test files
    for file_info in test_files:
        src = file_info['path']
        dst = test_path / file_info['name']
        
        if method == 'symlink':
            dst.symlink_to(src.resolve())
        elif method == 'copy':
            shutil.copy2(src, dst)
    
    print(f"✓ Processed {len(test_files)} test files")


def save_split_info(output_dir, train_files, test_files, args):
    """Save split information for reproducibility"""
    
    output_path = Path(output_dir)
    
    split_info = {
        'method': args.method,
        'test_ratio': args.test_ratio,
        'seed': args.seed,
        'n_train': len(train_files),
        'n_test': len(test_files),
        'train_files': [
            {
                'name': f['name'],
                'source': f['source'],
                'path': str(f['path'])
            } for f in train_files
        ],
        'test_files': [
            {
                'name': f['name'],
                'source': f['source'],
                'path': str(f['path'])
            } for f in test_files
        ]
    }
    
    # Save as JSON
    info_file = output_path / 'split_info.json'
    with open(info_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"✓ Saved split info to {info_file}")
    
    # Also save simple text lists
    train_list = output_path / 'train_files.txt'
    with open(train_list, 'w') as f:
        for file_info in train_files:
            f.write(f"{file_info['name']}\n")
    
    test_list = output_path / 'test_files.txt'
    with open(test_list, 'w') as f:
        for file_info in test_files:
            f.write(f"{file_info['name']}\n")
    
    print(f"✓ Saved file lists to {train_list} and {test_list}")


def main():
    parser = argparse.ArgumentParser(
        description='Combine training labels and create train/test split'
    )
    
    parser.add_argument(
        '--base_dir',
        type=str,
        default='/training_labels',
        help='Base directory containing subdirectories (default: /training_labels)'
    )
    
    parser.add_argument(
        '--subdirs',
        type=str,
        nargs='+',
        default=['t1w', 't2w-cor', 't2w-tra'],
        help='Subdirectories to combine (default: t1w t2w-cor t2w-tra)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/tmp/training_split',
        help='Output directory for train/test split (default: /tmp/training_split)'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        choices=['symlink', 'copy'],
        default='copy',
        help='Method to populate directories: symlink (efficient) or copy (safer)'
    )
    
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.2,
        help='Ratio of files to use for testing (default: 0.2 = 20%%)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("Creating Train/Test Split for SynthSeg Training")
    print(f"{'='*60}\n")
    
    # Step 1: Collect all files
    all_files = collect_all_labels(args.base_dir, args.subdirs)
    if len(all_files) == 0:
        print("No files found! Check your base_dir and subdirs.")
        return

    # Step 2: Create split
    train_files, test_files = create_split(all_files, args.test_ratio, args.seed)
    
    # Step 3: Create directories
    train_path, test_path = create_directories(args.output_dir, args.method)
    
    if train_path is None:
        return
    
    # Step 4: Populate directories
    populate_directories(train_files, test_files, train_path, test_path, args.method)
    
    # Step 5: Save split information
    save_split_info(args.output_dir, train_files, test_files, args)

if __name__ == '__main__':
    main()

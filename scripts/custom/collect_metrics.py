#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def compute_extended_summary(folders, out_dir=None, hd_threshold=30.0):
    """
    Reads 'metrics.csv' from each specified folder, ignores non-metric columns 
    (like subject IDs or label numbers), filters out HD95 outliers/artifacts,
    and writes a cleaned summary CSV.
    """
    epoch_rows = []
    
    # Non-metric metadata columns to explicitly exclude from summary calculations
    IGNORE_COLS = {'subject', 'label', 'structure', 'id', 'unnamed: 0'}

    for folder in folders:
        folder = Path(folder)
        csv_path = folder / 'metrics.csv'
        
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found. Skipping.")
            continue

        df = pd.read_csv(csv_path)

        # Select numeric metric columns while excluding non-metric identifier columns
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns 
            if c.lower().strip() not in IGNORE_COLS
        ]

        row_dict = {
            'folder': folder.name,
            'path': str(folder)
        }

        for col in numeric_cols:
            series = df[col].copy()

            # --- FILTERING RULE: Handle HD95/Distance Metrics ---
            is_distance_col = any(hd_key in col.lower() for hd_key in ['hd', 'hausdorff', 'distance'])
            
            if is_distance_col:
                # 1. Look for a matching Dice column to drop distances where Dice == 0 (undetected)
                col_suffix = col.replace('hd95', '').replace('hausdorff_95', '').replace('mean_distance', '')
                matching_dice_cols = [
                    c for c in numeric_cols 
                    if 'dice' in c.lower() and (col_suffix in c or col_suffix == '')
                ]

                if matching_dice_cols:
                    dice_series = df[matching_dice_cols[0]]
                    series = series[dice_series > 0.0]

                # 2. Hard threshold cutoff: remove distance artifacts > 30mm (inter-hemispheric / bbox max)
                series = series[series <= hd_threshold]

            # Drop NaNs after filtering
            series = series.dropna()

            if series.empty:
                continue

            # Compute Summary Statistics
            mean_val = series.mean()
            std_val = series.std(ddof=1) if len(series) > 1 else 0.0
            median_val = series.median()
            q25_val = series.quantile(0.25)
            q75_val = series.quantile(0.75)
            iqr_val = q75_val - q25_val

            # Store in output dictionary
            row_dict[f'{col}_mean'] = mean_val
            row_dict[f'{col}_std'] = std_val
            row_dict[f'{col}_median'] = median_val
            row_dict[f'{col}_iqr'] = iqr_val
            row_dict[f'{col}_n_valid'] = len(series)
            row_dict[f'{col}_mean_std_str'] = f"{mean_val:.4f} ± {std_val:.4f}"
            row_dict[f'{col}_median_iqr_str'] = f"{median_val:.4f} [{q25_val:.4f}, {q75_val:.4f}]"

        epoch_rows.append(row_dict)

    if not epoch_rows:
        print("No valid metric files found to summarize.")
        return None

    summary_df = pd.DataFrame(epoch_rows)

    # Determine output path
    if out_dir is None:
        if len(folders) > 1:
            try:
                out_dir = os.path.commonpath([str(Path(f).resolve()) for f in folders])
            except ValueError:
                out_dir = str(Path(folders[0]).resolve().parent)
        else:
            out_dir = str(Path(folders[0]).resolve())

    out_path = Path(out_dir) / 'metrics_summary_stats.csv'
    summary_df.to_csv(out_path, index=False)
    print(f"\nWrote cleaned per-epoch summary stats to:\n  {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Collect and summarize medical segmentation metrics.")
    parser.add_argument('folders', nargs='+', help="List of run/epoch folders containing 'metrics.csv'")
    parser.add_argument('--out_dir', type=str, default=None, help="Directory to save 'metrics_summary_stats.csv'")
    parser.add_argument('--hd_threshold', type=float, default=30.0, help="Maximum allowed HD95 value in mm (default: 30.0)")

    args = parser.parse_args()
    compute_extended_summary(args.folders, out_dir=args.out_dir, hd_threshold=args.hd_threshold)


if __name__ == '__main__':
    main()

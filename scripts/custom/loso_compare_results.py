"""
LOSO Label Quality Comparison

Analyzes and visualizes LOSO cross-validation results to identify outlier labels.

Usage:
    python scripts/custom/loso_compare_results.py \
        --output_dir /home/althause/data/loso_label_quality
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def load_results(output_dir):
    """Load all LOSO results from fold directories."""
    all_results = []
    fold_dirs = sorted(Path(output_dir).glob('fold_*'))

    for fold_dir in fold_dirs:
        results_path = fold_dir / 'results.json'
        if not results_path.exists():
            continue
        with open(results_path) as f:
            fold_results = json.load(f)
        all_results.extend(fold_results)

    return pd.DataFrame(all_results) if all_results else pd.DataFrame()


def print_quality_summary(df, output_dir):
    """Print detailed quality summary."""
    if df.empty:
        print("No results found.")
        return

    # Per-subject summary
    subject_summary = df.groupby('subject_id').agg({
        'dice': ['mean', 'std', 'min', 'max', 'count'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'iou': ['mean', 'std'],
    }).round(4)

    print("\n" + "="*90)
    print("PER-SUBJECT QUALITY SUMMARY")
    print("="*90)
    print(subject_summary.to_string())

    # Per-hemisphere summary
    hemi_summary = df.groupby('hemisphere').agg({
        'dice': ['mean', 'std', 'min', 'max'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'iou': ['mean', 'std'],
    }).round(4)

    print("\n" + "="*90)
    print("PER-HEMISPHERE SUMMARY")
    print("="*90)
    print(hemi_summary.to_string())

    # Overall statistics
    print("\n" + "="*90)
    print("OVERALL STATISTICS")
    print("="*90)
    for col in ['dice', 'precision', 'recall', 'iou']:
        if col in df.columns:
            print(f"{col.upper():12s} mean={df[col].mean():.4f}  std={df[col].std():.4f}  "
                  f"min={df[col].min():.4f}  max={df[col].max():.4f}")

    # Save summaries
    subject_summary.to_csv(os.path.join(output_dir, 'subject_summary.csv'))
    hemi_summary.to_csv(os.path.join(output_dir, 'hemisphere_summary.csv'))
    print(f"\nSummaries saved to {output_dir}")


def create_visualizations(df, output_dir):
    """Create comparison plots."""
    if df.empty:
        print("No data for visualization.")
        return

    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)

    # 1. Dice distribution per subject
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    subject_dice = df.groupby('subject_id')['dice'].mean().sort_values()
    axes[0].barh(range(len(subject_dice)), subject_dice.values, color='steelblue')
    axes[0].set_yticks(range(len(subject_dice)))
    axes[0].set_yticklabels(subject_dice.index, fontsize=8)
    axes[0].set_xlabel('Mean Dice')
    axes[0].set_title('Mean Dice per Subject (sorted)')
    axes[0].axvline(x=0.50, color='red', linestyle='--', label='Threshold (0.50)')
    axes[0].legend()
    axes[0].grid(axis='x', alpha=0.3)

    # 2. Recall distribution per subject
    subject_recall = df.groupby('subject_id')['recall'].mean().sort_values()
    axes[1].barh(range(len(subject_recall)), subject_recall.values, color='coral')
    axes[1].set_yticks(range(len(subject_recall)))
    axes[1].set_yticklabels(subject_recall.index, fontsize=8)
    axes[1].set_xlabel('Mean Recall')
    axes[1].set_title('Mean Recall per Subject (sorted)')
    axes[1].axvline(x=0.45, color='red', linestyle='--', label='Threshold (0.45)')
    axes[1].legend()
    axes[1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', '01_dice_recall_per_subject.png'), dpi=150)
    plt.close()

    # 3. Scatter: Dice vs Recall
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df['recall'], df['dice'], c=df['iou'], cmap='viridis', s=50, alpha=0.6)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Dice')
    ax.set_title('Dice vs Recall (colored by IoU)')
    ax.axvline(x=0.45, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=0.50, color='red', linestyle='--', alpha=0.5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('IoU')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', '02_dice_vs_recall.png'), dpi=150)
    plt.close()

    # 4. Box plots by hemisphere
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics = ['dice', 'precision', 'recall', 'iou']
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        df.boxplot(column=metric, by='hemisphere', ax=ax)
        ax.set_title(f'{metric.upper()} by Hemisphere')
        ax.set_xlabel('Hemisphere')
        ax.set_ylabel(metric.upper())
        ax.grid(alpha=0.3)
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', '03_metrics_by_hemisphere.png'), dpi=150)
    plt.close()

    # 5. Distribution histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        ax.hist(df[metric], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel(metric.upper())
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {metric.upper()}')
        ax.axvline(x=df[metric].mean(), color='red', linestyle='--', label=f'Mean: {df[metric].mean():.3f}')
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', '04_metric_distributions.png'), dpi=150)
    plt.close()

    print(f"Plots saved to {os.path.join(output_dir, 'plots')}")


def identify_outliers(df, dice_threshold=0.50, recall_threshold=0.45):
    """Identify and report outlier subjects."""
    print("\n" + "="*90)
    print("OUTLIER DETECTION")
    print("="*90)

    low_dice = df[df['dice'] < dice_threshold]
    low_recall = df[df['recall'] < recall_threshold]

    if not low_dice.empty:
        print(f"\n❌ LOW DICE (< {dice_threshold}):")
        for subject in low_dice['subject_id'].unique():
            mean_dice = low_dice[low_dice['subject_id'] == subject]['dice'].mean()
            print(f"   {subject}: {mean_dice:.4f}")

    if not low_recall.empty:
        print(f"\n⚠️  LOW RECALL (< {recall_threshold}):")
        for subject in low_recall['subject_id'].unique():
            mean_recall = low_recall[low_recall['subject_id'] == subject]['recall'].mean()
            print(f"   {subject}: {mean_recall:.4f}")

    # Find subjects with both metrics below threshold
    problem_subjects = set(low_dice['subject_id'].unique()) & set(low_recall['subject_id'].unique())
    if problem_subjects:
        print(f"\n🔴 HIGH PRIORITY FOR REVIEW (both Dice and Recall low):")
        for subject in sorted(problem_subjects):
            s_df = df[df['subject_id'] == subject]
            print(f"   {subject}: Dice={s_df['dice'].mean():.4f}, Recall={s_df['recall'].mean():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='/home/althause/data/loso_label_quality')
    parser.add_argument('--dice_threshold', type=float, default=0.50)
    parser.add_argument('--recall_threshold', type=float, default=0.45)
    args = parser.parse_args()

    print(f"Loading LOSO results from {args.output_dir}...")
    df = load_results(args.output_dir)

    if df.empty:
        print("ERROR: No results found.")
        return

    print(f"Loaded {len(df)} evaluation results from {len(df['subject_id'].unique())} subjects")

    # Print summaries
    print_quality_summary(df, args.output_dir)

    # Create visualizations
    create_visualizations(df, args.output_dir)

    # Identify outliers
    identify_outliers(df, args.dice_threshold, args.recall_threshold)

    print("\n" + "="*90)
    print("Analysis complete.")


if __name__ == '__main__':
    main()

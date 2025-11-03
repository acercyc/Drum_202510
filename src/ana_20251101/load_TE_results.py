"""
Load and analyze Transfer Entropy results from TE_results_all.csv

This script loads the TE computation results and provides functions for analysis.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


def load_te_results(csv_path: str = "TE_results_all.csv") -> pd.DataFrame:
    """
    Load Transfer Entropy results from CSV file.
    
    Args:
        csv_path: Path to the TE results CSV file
        
    Returns:
        DataFrame with TE results and associated metadata
    """
    csv_file = Path(csv_path)
    
    if not csv_file.exists():
        # Try relative to script directory
        script_dir = Path(__file__).parent
        csv_file = script_dir / csv_path
        if not csv_file.exists():
            raise FileNotFoundError(f"TE results file not found: {csv_path}")
    
    print(f"Loading TE results from: {csv_file}")
    
    # Read CSV - handle potential header issues
    # First, read without assuming header to check structure
    df_raw = pd.read_csv(csv_file, header=0)
    
    # The CSV has a malformed first header row - use only the first 9 columns as expected
    expected_cols = ['month', 'subject', 'date', 'ni_file', 'TE', 'name', 'Q1', 'Q2', 'datetime']
    
    # If we have the expected columns, use them
    if all(col in df_raw.columns for col in expected_cols[:9]):
        df = df_raw[expected_cols].copy()
    else:
        # Fallback: take first 9 columns and rename them
        df = df_raw.iloc[:, :9].copy()
        df.columns = expected_cols
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Convert TE to numeric if it's not already
    if 'TE' in df.columns:
        df['TE'] = pd.to_numeric(df['TE'], errors='coerce')
    
    # Convert Q1 and Q2 to numeric
    for col in ['Q1', 'Q2']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert date to string for easier filtering
    if 'date' in df.columns:
        df['date'] = df['date'].astype(str)
    
    # Convert subject to string for consistency
    if 'subject' in df.columns:
        df['subject'] = df['subject'].astype(str)
    
    return df


def explore_results(df: pd.DataFrame):
    """
    Explore and display summary statistics of the TE results.
    
    Args:
        df: DataFrame with TE results
    """
    print("\n" + "=" * 80)
    print("TE RESULTS EXPLORATION")
    print("=" * 80)
    
    # Basic statistics
    print(f"\nTotal records: {len(df)}")
    print(f"Unique months: {df['month'].nunique()} ({', '.join(sorted(df['month'].unique()))})")
    print(f"Unique subjects: {df['subject'].nunique()} ({', '.join(sorted(df['subject'].astype(str).unique()))})")
    print(f"Unique dates: {df['date'].nunique()}")
    
    # TE statistics
    if 'TE' in df.columns:
        valid_te = df['TE'].dropna()
        print(f"\nTE Statistics:")
        print(f"  Valid TE values: {len(valid_te)} / {len(df)}")
        print(f"  Mean: {valid_te.mean():.6f}")
        print(f"  Std: {valid_te.std():.6f}")
        print(f"  Min: {valid_te.min():.6f}")
        print(f"  Max: {valid_te.max():.6f}")
        print(f"  Median: {valid_te.median():.6f}")
    
    # Statistics by month
    if 'month' in df.columns and 'TE' in df.columns:
        print("\nTE Statistics by Month:")
        month_stats = df.groupby('month')['TE'].agg(['count', 'mean', 'std', 'min', 'max'])
        print(month_stats)
    
    # Statistics by subject
    if 'subject' in df.columns and 'TE' in df.columns:
        print("\nTE Statistics by Subject:")
        subject_stats = df.groupby('subject')['TE'].agg(['count', 'mean', 'std', 'min', 'max'])
        print(subject_stats)
    
    # SoA statistics (if available)
    if 'Q1' in df.columns:
        print("\nSoA Q1 Statistics:")
        q1_valid = df['Q1'].dropna()
        print(f"  Valid Q1 values: {len(q1_valid)} / {len(df)}")
        print(f"  Mean: {q1_valid.mean():.2f}")
        print(f"  Std: {q1_valid.std():.2f}")
        print(f"  Range: {q1_valid.min():.0f} - {q1_valid.max():.0f}")
    
    if 'Q2' in df.columns:
        print("\nSoA Q2 Statistics:")
        q2_valid = df['Q2'].dropna()
        print(f"  Valid Q2 values: {len(q2_valid)} / {len(df)}")
        print(f"  Mean: {q2_valid.mean():.2f}")
        print(f"  Std: {q2_valid.std():.2f}")
        print(f"  Range: {q2_valid.min():.0f} - {q2_valid.max():.0f}")


def plot_te_overview(df: pd.DataFrame, save_plot: bool = False):
    """
    Create overview plots of TE results.
    
    Args:
        df: DataFrame with TE results
        save_plot: Whether to save plots
    """
    if 'TE' not in df.columns:
        print("No TE column found in data")
        return
    
    valid_df = df.dropna(subset=['TE'])
    
    # Plot 1: TE distribution histogram
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(valid_df['TE'], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Transfer Entropy')
    plt.ylabel('Frequency')
    plt.title('Distribution of Transfer Entropy Values')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: TE over time/index
    plt.subplot(1, 2, 2)
    plt.plot(valid_df.index, valid_df['TE'], 'o', markersize=3, alpha=0.6)
    plt.xlabel('Index')
    plt.ylabel('Transfer Entropy')
    plt.title('Transfer Entropy Across All Sessions')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        output_path = "TE_overview.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nOverview plot saved to: {output_path}")
    else:
        plt.show()


def plot_te_by_group(df: pd.DataFrame, group_by: str = 'subject', save_plot: bool = False):
    """
    Plot TE results grouped by month, subject, or date.
    
    Args:
        df: DataFrame with TE results
        group_by: Column to group by ('month', 'subject', or 'date')
        save_plot: Whether to save plots
    """
    if group_by not in df.columns:
        print(f"Column '{group_by}' not found in data")
        return
    
    if 'TE' not in df.columns:
        print("No TE column found in data")
        return
    
    valid_df = df.dropna(subset=['TE'])
    
    # Box plot
    plt.figure(figsize=(14, 6))
    
    groups = sorted(valid_df[group_by].unique())
    te_by_group = [valid_df[valid_df[group_by] == g]['TE'].values for g in groups]
    
    plt.boxplot(te_by_group, labels=[str(g) for g in groups])
    plt.xlabel(group_by.capitalize())
    plt.ylabel('Transfer Entropy')
    plt.title(f'Transfer Entropy Distribution by {group_by.capitalize()}')
    plt.xticks(rotation=45 if group_by == 'date' else 0)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_plot:
        output_path = f"TE_by_{group_by}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()


def plot_te_vs_soa(df: pd.DataFrame, save_plot: bool = False):
    """
    Plot TE vs SoA ratings (Q1, Q2).
    
    Args:
        df: DataFrame with TE results
        save_plot: Whether to save plots
    """
    if 'TE' not in df.columns:
        print("No TE column found")
        return
    
    # Check for SoA columns
    has_q1 = 'Q1' in df.columns
    has_q2 = 'Q2' in df.columns
    
    if not (has_q1 or has_q2):
        print("No SoA columns (Q1, Q2) found")
        return
    
    fig, axes = plt.subplots(1, 2 if (has_q1 and has_q2) else 1, figsize=(14, 5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    idx = 0
    
    if has_q1:
        valid_data = df.dropna(subset=['TE', 'Q1'])
        if len(valid_data) > 0:
            axes[idx].scatter(valid_data['Q1'], valid_data['TE'], alpha=0.6, s=50)
            axes[idx].set_xlabel('Q1 (SoA Rating)')
            axes[idx].set_ylabel('Transfer Entropy')
            axes[idx].set_title('TE vs Q1')
            axes[idx].grid(True, alpha=0.3)
            
            # Add correlation
            corr = valid_data['Q1'].corr(valid_data['TE'])
            axes[idx].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                          transform=axes[idx].transAxes, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            idx += 1
    
    if has_q2:
        valid_data = df.dropna(subset=['TE', 'Q2'])
        if len(valid_data) > 0:
            axes[idx].scatter(valid_data['Q2'], valid_data['TE'], alpha=0.6, s=50, color='orange')
            axes[idx].set_xlabel('Q2 (SoA Rating)')
            axes[idx].set_ylabel('Transfer Entropy')
            axes[idx].set_title('TE vs Q2')
            axes[idx].grid(True, alpha=0.3)
            
            # Add correlation
            corr = valid_data['Q2'].corr(valid_data['TE'])
            axes[idx].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                          transform=axes[idx].transAxes, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_plot:
        output_path = "TE_vs_SoA.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()


def filter_results(df: pd.DataFrame, 
                   month: str = None, 
                   subject: str = None, 
                   date: str = None,
                   min_te: float = None,
                   max_te: float = None) -> pd.DataFrame:
    """
    Filter TE results by various criteria.
    
    Args:
        df: DataFrame with TE results
        month: Filter by month
        subject: Filter by subject
        date: Filter by date
        min_te: Minimum TE value
        max_te: Maximum TE value
        
    Returns:
        Filtered DataFrame
    """
    filtered = df.copy()
    
    if month is not None:
        filtered = filtered[filtered['month'] == month]
    
    if subject is not None:
        filtered = filtered[filtered['subject'].astype(str) == str(subject)]
    
    if date is not None:
        filtered = filtered[filtered['date'].astype(str) == str(date)]
    
    if min_te is not None:
        filtered = filtered[filtered['TE'] >= min_te]
    
    if max_te is not None:
        filtered = filtered[filtered['TE'] <= max_te]
    
    return filtered


def main():
    """Main function to load and explore TE results."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and analyze TE results")
    parser.add_argument("--csv", type=str, default="TE_results_all.csv", 
                       help="Path to TE results CSV file")
    parser.add_argument("--explore", action="store_true", 
                       help="Show exploration statistics")
    parser.add_argument("--plot-overview", action="store_true", 
                       help="Plot TE overview")
    parser.add_argument("--plot-by", type=str, choices=['month', 'subject', 'date'],
                       help="Plot TE grouped by month/subject/date")
    parser.add_argument("--plot-soa", action="store_true",
                       help="Plot TE vs SoA ratings")
    parser.add_argument("--save", action="store_true",
                       help="Save plots instead of showing")
    parser.add_argument("--filter-month", type=str,
                       help="Filter by month (e.g., 5月)")
    parser.add_argument("--filter-subject", type=str,
                       help="Filter by subject (e.g., 3)")
    parser.add_argument("--filter-date", type=str,
                       help="Filter by date (e.g., 20250507)")
    
    args = parser.parse_args()
    
    # Load data
    df = load_te_results(args.csv)
    
    # Apply filters if specified
    if args.filter_month or args.filter_subject or args.filter_date:
        df = filter_results(df, 
                           month=args.filter_month,
                           subject=args.filter_subject,
                           date=args.filter_date)
        print(f"\nAfter filtering: {len(df)} rows")
    
    # Exploration
    if args.explore:
        explore_results(df)
    
    # Plots
    if args.plot_overview:
        plot_te_overview(df, save_plot=args.save)
    
    if args.plot_by:
        plot_te_by_group(df, group_by=args.plot_by, save_plot=args.save)
    
    if args.plot_soa:
        plot_te_vs_soa(df, save_plot=args.save)
    
    # If no specific action, just show basic info
    if not (args.explore or args.plot_overview or args.plot_by or args.plot_soa):
        print("\n" + "=" * 80)
        print("Basic Information")
        print("=" * 80)
        print(f"\nDataFrame shape: {df.shape}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())
        print("\nUse --explore, --plot-overview, --plot-by, or --plot-soa for more analysis")


if __name__ == "__main__":
    main()


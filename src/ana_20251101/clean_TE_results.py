"""
Remove rows with missing data from TE_results_all.csv and create a clean file.

This script removes rows that have missing values in essential columns:
- TE (Transfer Entropy value)
- name (participant name)
- Q1, Q2 (SoA ratings)
- datetime (session timestamp)
"""
import pandas as pd
import argparse
from pathlib import Path


def clean_te_results(input_file: str = "TE_results_all.csv",
                    output_file: str = "TE_results_all_clean.csv",
                    required_cols: list = None):
    """
    Remove rows with missing data from TE results CSV.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output clean CSV file
        required_cols: List of columns that must not be missing.
                     If None, uses ['TE', 'name', 'Q1', 'Q2', 'datetime']
    
    Returns:
        Cleaned DataFrame
    """
    if required_cols is None:
        required_cols = ['TE', 'name', 'Q1', 'Q2', 'datetime']
    
    # Read CSV
    print(f"Reading: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"Original: {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Check missing data
    print("\nMissing data summary:")
    missing_summary = df.isna().sum()
    for col in required_cols:
        if col in df.columns:
            print(f"  {col}: {missing_summary[col]} missing")
    
    # Remove rows with missing required columns
    df_clean = df.dropna(subset=required_cols)
    
    removed_count = len(df) - len(df_clean)
    print(f"\nRemoved {removed_count} rows with missing data")
    print(f"Clean data: {len(df_clean)} rows")
    
    # Show removed rows if any
    if removed_count > 0:
        missing_mask = df[required_cols].isna().any(axis=1)
        removed_rows = df[missing_mask]
        print(f"\nRemoved rows:")
        print(removed_rows[['month', 'subject', 'date', 'ni_file', 'TE', 'name', 'Q1', 'Q2']].to_string())
    
    # Save clean file
    df_clean.to_csv(output_file, index=False)
    print(f"\nClean CSV saved to: {output_file}")
    
    # Summary statistics
    print("\nClean data statistics:")
    print(f"  Total rows: {len(df_clean)}")
    print(f"  Valid TE values: {df_clean['TE'].notna().sum()}")
    print(f"  Unique subjects: {df_clean['subject'].nunique()}")
    print(f"  Unique dates: {df_clean['date'].nunique()}")
    print(f"  Unique months: {df_clean['month'].nunique()}")
    
    if 'TE' in df_clean.columns:
        valid_te = df_clean['TE'].dropna()
        print(f"\nTE Statistics:")
        print(f"  Mean: {valid_te.mean():.6f}")
        print(f"  Std: {valid_te.std():.6f}")
        print(f"  Min: {valid_te.min():.6f}")
        print(f"  Max: {valid_te.max():.6f}")
    
    return df_clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean TE results CSV by removing rows with missing data")
    parser.add_argument("--input", type=str, default="TE_results_all.csv",
                       help="Input CSV file (default: TE_results_all.csv)")
    parser.add_argument("--output", type=str, default="TE_results_all_clean.csv",
                       help="Output clean CSV file (default: TE_results_all_clean.csv)")
    parser.add_argument("--cols", type=str, nargs="+",
                       help="Required columns (default: TE name Q1 Q2 datetime)")
    
    args = parser.parse_args()
    
    required_cols = args.cols if args.cols else None
    
    clean_te_results(args.input, args.output, required_cols)


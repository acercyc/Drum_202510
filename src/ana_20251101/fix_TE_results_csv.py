"""
Fix the corrupted TE_results_all.csv file.

The header row has extra columns that came from a data row being appended incorrectly.
This script cleans the CSV and creates a properly formatted version.
"""
import pandas as pd
import numpy as np
from pathlib import Path


def fix_te_results_csv(input_file: str = "TE_results_all.csv", 
                       output_file: str = "TE_results_all_fixed.csv"):
    """
    Fix the corrupted CSV file by extracting only valid columns and recovering
    any data that ended up in the wrong columns.
    
    Args:
        input_file: Path to corrupted CSV
        output_file: Path to save fixed CSV
    """
    print(f"Reading corrupted CSV: {input_file}")
    df_raw = pd.read_csv(input_file)
    
    print(f"Original shape: {df_raw.shape}")
    print(f"Original columns: {list(df_raw.columns)}")
    
    # Expected valid columns
    valid_cols = ['month', 'subject', 'date', 'ni_file', 'TE', 'name', 'Q1', 'Q2', 'datetime']
    
    # Extract only valid columns
    df_clean = df_raw[valid_cols].copy()
    
    # Try to recover data from corrupted columns
    # Check if 'datetimenakamura_0521' has datetime values that should be in 'datetime'
    if 'datetimenakamura_0521' in df_raw.columns:
        mask = df_clean['datetime'].isna() & df_raw['datetimenakamura_0521'].notna()
        if mask.sum() > 0:
            print(f"\nRecovering {mask.sum()} datetime values from 'datetimenakamura_0521' column")
            df_clean.loc[mask, 'datetime'] = df_raw.loc[mask, 'datetimenakamura_0521']
    
    # Check other weird columns for recoverable data
    # '2025-09-03 10:11:31' might have datetime values
    datetime_cols = [col for col in df_raw.columns if '2025-' in str(col) and ':' in str(col)]
    for col in datetime_cols:
        mask = df_clean['datetime'].isna() & df_raw[col].notna()
        if mask.sum() > 0:
            print(f"Recovering {mask.sum()} datetime values from '{col}' column")
            df_clean.loc[mask, 'datetime'] = df_raw.loc[mask, col]
    
    # Check for name values in weird columns
    name_cols = [col for col in df_raw.columns if col not in valid_cols and df_raw[col].dtype == 'object']
    for col in name_cols:
        if col == 'datetimenakamura_0521':  # Skip datetime-like columns
            continue
        mask = df_clean['name'].isna() & df_raw[col].notna()
        if mask.sum() > 0:
            # Check if values look like names (not datetime, not numeric)
            sample_values = df_raw.loc[mask, col].head(5)
            if all(isinstance(v, str) and not v.replace('.', '').replace('-', '').isdigit() 
                   for v in sample_values if pd.notna(v)):
                print(f"Recovering {mask.sum()} name values from '{col}' column")
                df_clean.loc[mask, 'name'] = df_raw.loc[mask, col]
    
    # Check for Q1/Q2 values in numeric columns
    numeric_cols = [col for col in df_raw.columns 
                   if col not in valid_cols and pd.api.types.is_numeric_dtype(df_raw[col])]
    for col in numeric_cols:
        mask_q1 = df_clean['Q1'].isna() & df_raw[col].notna()
        mask_q2 = df_clean['Q2'].isna() & df_raw[col].notna()
        
        if mask_q1.sum() > 0:
            # Check if values are in reasonable Q1 range (0-100)
            values = df_raw.loc[mask_q1, col]
            if values.between(0, 100).all():
                print(f"Recovering {mask_q1.sum()} Q1 values from '{col}' column")
                df_clean.loc[mask_q1, 'Q1'] = df_raw.loc[mask_q1, col]
        
        if mask_q2.sum() > 0:
            values = df_raw.loc[mask_q2, col]
            if values.between(0, 100).all():
                print(f"Recovering {mask_q2.sum()} Q2 values from '{col}' column")
                df_clean.loc[mask_q2, 'Q2'] = df_raw.loc[mask_q2, col]
    
    # Ensure proper data types
    if 'TE' in df_clean.columns:
        df_clean['TE'] = pd.to_numeric(df_clean['TE'], errors='coerce')
    
    for col in ['Q1', 'Q2']:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    if 'date' in df_clean.columns:
        df_clean['date'] = df_clean['date'].astype(str)
    
    if 'subject' in df_clean.columns:
        df_clean['subject'] = df_clean['subject'].astype(str)
    
    # Save cleaned CSV
    print(f"\nSaving fixed CSV to: {output_file}")
    df_clean.to_csv(output_file, index=False)
    
    print(f"\nFixed CSV shape: {df_clean.shape}")
    print(f"Fixed CSV columns: {list(df_clean.columns)}")
    print(f"\nMissing values:")
    print(df_clean.isna().sum())
    
    # Summary statistics
    print(f"\nSummary:")
    print(f"  Total rows: {len(df_clean)}")
    print(f"  Valid TE values: {df_clean['TE'].notna().sum()}")
    print(f"  Valid Q1 values: {df_clean['Q1'].notna().sum()}")
    print(f"  Valid Q2 values: {df_clean['Q2'].notna().sum()}")
    print(f"  Valid datetime values: {df_clean['datetime'].notna().sum()}")
    
    return df_clean


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix corrupted TE results CSV")
    parser.add_argument("--input", type=str, default="TE_results_all.csv",
                       help="Input CSV file (corrupted)")
    parser.add_argument("--output", type=str, default="TE_results_all_fixed.csv",
                       help="Output CSV file (fixed)")
    parser.add_argument("--replace", action="store_true",
                       help="Replace original file with fixed version")
    
    args = parser.parse_args()
    
    df_fixed = fix_te_results_csv(args.input, args.output)
    
    if args.replace:
        import shutil
        backup_file = args.input + ".backup"
        shutil.copy(args.input, backup_file)
        print(f"\nBackup saved to: {backup_file}")
        shutil.copy(args.output, args.input)
        print(f"Original file replaced with fixed version")


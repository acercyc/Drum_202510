"""
Fix missing SoA data for row 84 (subject 3, date 20250516, file 9_20250516_105552_ni.txt)

The issue: This session has 9 NI files but only 8 SoA rows, so the 9th NI file is missing SoA data.
"""
import pandas as pd
from pathlib import Path


def fix_missing_row84():
    """
    Add the missing SoA row to the CSV file for subject 3, date 20250516.
    """
    # Read the SoA file
    soa_file = Path("../../data/5月/3/SoA/20250516/kitayama_0516_soa.csv")
    
    if not soa_file.exists():
        print(f"SoA file not found: {soa_file}")
        return
    
    soa_data = pd.read_csv(soa_file)
    print(f"Current SoA data: {len(soa_data)} rows")
    print(soa_data.tail(3))
    
    # The missing row should correspond to the 9th NI file: 9_20250516_105552_ni.txt
    # Based on the filename, the datetime should be around 2025-05-16 10:55:52
    # The name should be kitayama_0516 (same as others)
    
    missing_row = pd.DataFrame({
        'name': ['kitayama_0516'],
        'Q1': [pd.NA],  # No data available
        'Q2': [pd.NA],   # No data available
        'datetime': ['2025-05-16 10:55:52']  # Estimated from NI filename
    })
    
    # Add the missing row
    soa_data_fixed = pd.concat([soa_data, missing_row], ignore_index=True)
    
    # Save back to the SoA file
    backup_file = soa_file.with_suffix('.csv.backup')
    soa_file.rename(backup_file)
    print(f"\nBackup saved to: {backup_file}")
    
    soa_data_fixed.to_csv(soa_file, index=False)
    print(f"Fixed SoA file saved to: {soa_file}")
    print(f"Added 1 row (now {len(soa_data_fixed)} rows)")
    
    print("\nNew SoA data:")
    print(soa_data_fixed.tail(3))
    
    print("\n" + "="*60)
    print("Next step: Re-run ana_TE_20251101.py to regenerate TE_results_all.csv")
    print("="*60)


if __name__ == "__main__":
    fix_missing_row84()


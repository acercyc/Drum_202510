"""
Generate a comprehensive CSV inventory of all data files.

This script creates a CSV file containing all information about data files:
- Month, Subject, Date
- NI file paths and counts
- SoA file paths
- File sizes and other metadata
"""
import pandas as pd
from pathlib import Path
from locate_data_files import find_all_data_groups, default_base_dir


def generate_inventory_csv(output_file: str = "data_inventory.csv"):
    """
    Generate a comprehensive CSV inventory of all data files.
    
    Args:
        output_file: Path to output CSV file
    """
    print("Generating data inventory...")
    
    # Find all data groups
    base_dir = default_base_dir()
    groups = find_all_data_groups(base_dir)
    
    print(f"Found {len(groups)} data file groups")
    
    # Build inventory records
    records = []
    
    for group in groups:
        # Basic information
        record = {
            'month': group.month,
            'subject': group.subject,
            'date': group.date,
            'ni_file_count': len(group.ni_files),
            'has_soa_csv': group.soa_csv is not None and group.soa_csv.exists(),
            'has_soa_xlsx': group.soa_xlsx is not None and group.soa_xlsx.exists(),
        }
        
        # NI file information
        if group.ni_files:
            # List all NI files (semicolon-separated)
            ni_file_names = [f.name for f in group.ni_files]
            record['ni_files'] = '; '.join(ni_file_names)
            
            # First and last NI file paths
            record['first_ni_file'] = str(group.ni_files[0])
            record['last_ni_file'] = str(group.ni_files[-1])
            
            # Total size of NI files (MB)
            total_size = sum(f.stat().st_size for f in group.ni_files if f.exists())
            record['total_ni_size_mb'] = round(total_size / (1024 * 1024), 2)
            
            # Average file size
            record['avg_ni_size_mb'] = round(total_size / len(group.ni_files) / (1024 * 1024), 2) if group.ni_files else 0
        else:
            record['ni_files'] = ''
            record['first_ni_file'] = ''
            record['last_ni_file'] = ''
            record['total_ni_size_mb'] = 0
            record['avg_ni_size_mb'] = 0
        
        # SoA file information
        if group.soa_csv and group.soa_csv.exists():
            record['soa_csv_path'] = str(group.soa_csv)
            record['soa_csv_size_kb'] = round(group.soa_csv.stat().st_size / 1024, 2)
            
            # Try to get row count from CSV
            try:
                soa_df = pd.read_csv(group.soa_csv)
                record['soa_csv_rows'] = len(soa_df)
                record['soa_csv_columns'] = '; '.join(soa_df.columns.tolist())
            except Exception as e:
                record['soa_csv_rows'] = None
                record['soa_csv_columns'] = f'Error: {str(e)[:50]}'
        else:
            record['soa_csv_path'] = ''
            record['soa_csv_size_kb'] = 0
            record['soa_csv_rows'] = None
            record['soa_csv_columns'] = ''
        
        if group.soa_xlsx and group.soa_xlsx.exists():
            record['soa_xlsx_path'] = str(group.soa_xlsx)
            record['soa_xlsx_size_kb'] = round(group.soa_xlsx.stat().st_size / 1024, 2)
        else:
            record['soa_xlsx_path'] = ''
            record['soa_xlsx_size_kb'] = 0
        
        records.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Sort by month, subject, date
    df = df.sort_values(['month', 'subject', 'date']).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\nInventory saved to: {output_file}")
    print(f"Total records: {len(df)}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    print(f"\nTotal groups: {len(df)}")
    print(f"Groups with NI files: {len(df[df['ni_file_count'] > 0])}")
    print(f"Groups with SoA CSV: {len(df[df['has_soa_csv']])}")
    print(f"Groups with SoA XLSX: {len(df[df['has_soa_xlsx']])}")
    
    print(f"\nTotal NI files: {df['ni_file_count'].sum()}")
    print(f"Total NI data size: {df['total_ni_size_mb'].sum():.2f} MB")
    
    print(f"\nUnique months: {df['month'].nunique()} ({', '.join(sorted(df['month'].unique()))})")
    print(f"Unique subjects: {df['subject'].nunique()} ({', '.join(sorted(df['subject'].unique()))})")
    print(f"Unique dates: {df['date'].nunique()}")
    
    print("\nBy month:")
    month_summary = df.groupby('month').agg({
        'ni_file_count': 'sum',
        'subject': 'nunique',
        'date': 'nunique'
    }).rename(columns={'subject': 'subjects', 'date': 'dates'})
    print(month_summary)
    
    print("\nBy subject:")
    subject_summary = df.groupby('subject').agg({
        'ni_file_count': 'sum',
        'date': 'nunique',
        'total_ni_size_mb': 'sum'
    }).rename(columns={'date': 'dates'})
    print(subject_summary)
    
    # Show first few rows
    print("\n" + "=" * 80)
    print("SAMPLE DATA (first 5 rows)")
    print("=" * 80)
    print(df[['month', 'subject', 'date', 'ni_file_count', 'has_soa_csv', 'total_ni_size_mb']].head().to_string(index=False))
    
    return df


if __name__ == "__main__":
    import sys
    
    output_file = sys.argv[1] if len(sys.argv) > 1 else "data_inventory.csv"
    df = generate_inventory_csv(output_file)
    print(f"\n✓ Complete! Inventory saved to: {output_file}")






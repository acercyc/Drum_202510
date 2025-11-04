"""
Detailed report of NI file vs SoA row mismatches, showing file names.
"""
import pandas as pd
from pathlib import Path
from locate_data_files import find_all_data_groups, default_base_dir

def count_soa_rows(soa_csv_path: Path):
    """Count rows in SoA CSV, handling missing headers."""
    try:
        soa_data_temp = pd.read_csv(soa_csv_path, nrows=1)
        expected_cols = ['name', 'Q1', 'Q2', 'datetime']
        has_header = all(col in soa_data_temp.columns for col in expected_cols)
        
        if not has_header:
            soa_data = pd.read_csv(soa_csv_path, header=None)
        else:
            soa_data = pd.read_csv(soa_csv_path)
        
        return soa_data
    except Exception as e:
        return None

base = default_base_dir()
groups = find_all_data_groups(base)

print("=" * 80)
print("DETAILED MISMATCH REPORT")
print("=" * 80)

mismatches = []
for group in groups:
    ni_count = len(group.ni_files)
    
    if group.soa_csv and group.soa_csv.exists() and ni_count > 0:
        soa_data = count_soa_rows(group.soa_csv)
        if soa_data is not None:
            soa_row_count = len(soa_data)
            
            if ni_count != soa_row_count:
                mismatches.append({
                    'group': group,
                    'ni_count': ni_count,
                    'soa_row_count': soa_row_count,
                    'difference': ni_count - soa_row_count,
                    'soa_data': soa_data
                })

for m in mismatches:
    group = m['group']
    print(f"\n{'='*80}")
    print(f"Group: {group.month}/{group.subject}/{group.date}")
    print(f"NI files: {m['ni_count']}, SoA rows: {m['soa_row_count']}, Difference: {m['difference']:+d}")
    print(f"{'='*80}")
    
    print(f"\nNI Files ({m['ni_count']}):")
    for i, ni_file in enumerate(group.ni_files, 1):
        print(f"  {i:2d}. {ni_file.name}")
    
    print(f"\nSoA CSV Rows ({m['soa_row_count']}):")
    soa_data = m['soa_data']
    for i, row in soa_data.iterrows():
        name = row.get('name', row.iloc[0] if len(row) > 0 else 'N/A')
        q1 = row.get('Q1', row.iloc[1] if len(row) > 1 else 'N/A')
        q2 = row.get('Q2', row.iloc[2] if len(row) > 2 else 'N/A')
        dt = row.get('datetime', row.iloc[3] if len(row) > 3 else 'N/A')
        print(f"  {i+1:2d}. name={name}, Q1={q1}, Q2={q2}, datetime={dt}")
    
    if m['difference'] > 0:
        print(f"\n⚠️  Missing {m['difference']} SoA row(s) - there are more NI files than SoA entries")
    else:
        print(f"\n⚠️  Extra {abs(m['difference'])} SoA row(s) - there are more SoA entries than NI files")

print(f"\n\n{'='*80}")
print(f"TOTAL: {len(mismatches)} groups with mismatches")
print(f"{'='*80}")


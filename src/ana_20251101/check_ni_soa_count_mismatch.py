"""
Check for inconsistencies between number of NI files and SoA CSV rows.

For each data group, the number of NI files should match the number of rows in the SoA CSV file.
"""
import pandas as pd
from pathlib import Path
from locate_data_files import find_all_data_groups, default_base_dir

def count_soa_rows(soa_csv_path: Path):
    """Count rows in SoA CSV, handling missing headers."""
    try:
        # Try reading with header first
        soa_data_temp = pd.read_csv(soa_csv_path, nrows=1)
        
        # Check if header is missing
        expected_cols = ['name', 'Q1', 'Q2', 'datetime']
        has_header = all(col in soa_data_temp.columns for col in expected_cols)
        
        if not has_header:
            # File is missing header - read without header
            soa_data = pd.read_csv(soa_csv_path, header=None)
        else:
            # File has proper header - read normally
            soa_data = pd.read_csv(soa_csv_path)
        
        return len(soa_data)
    except Exception as e:
        return None, str(e)

# Main checking
base = default_base_dir()
groups = find_all_data_groups(base)

print("Checking for mismatches between NI file count and SoA CSV row count...\n")

mismatches = []
missing_soa = []
missing_ni = []

for group in groups:
    ni_count = len(group.ni_files)
    
    if group.soa_csv and group.soa_csv.exists():
        soa_row_count = count_soa_rows(group.soa_csv)
        
        if soa_row_count is None:
            missing_soa.append({
                'group': f"{group.month}/{group.subject}/{group.date}",
                'ni_count': ni_count,
                'soa_file': str(group.soa_csv),
                'issue': 'Could not read SoA CSV'
            })
        elif ni_count != soa_row_count:
            mismatches.append({
                'group': f"{group.month}/{group.subject}/{group.date}",
                'ni_count': ni_count,
                'soa_row_count': soa_row_count,
                'difference': ni_count - soa_row_count,
                'soa_file': str(group.soa_csv)
            })
    elif ni_count > 0:
        missing_ni.append({
            'group': f"{group.month}/{group.subject}/{group.date}",
            'ni_count': ni_count,
            'issue': 'No SoA CSV file found'
        })

# Report results
print("=" * 80)
print("MISMATCHES: NI file count ≠ SoA CSV row count")
print("=" * 80)
if mismatches:
    print(f"\nFound {len(mismatches)} groups with mismatches:\n")
    for m in mismatches:
        print(f"  Group: {m['group']}")
        print(f"    NI files: {m['ni_count']}")
        print(f"    SoA rows: {m['soa_row_count']}")
        print(f"    Difference: {m['difference']:+d} ({'more NI files' if m['difference'] > 0 else 'more SoA rows'})")
        print(f"    SoA file: {m['soa_file']}")
        print()
else:
    print("\n✓ No mismatches found! All groups have matching counts.")

print("\n" + "=" * 80)
print("MISSING SOA: Groups with NI files but no SoA CSV")
print("=" * 80)
if missing_ni:
    print(f"\nFound {len(missing_ni)} groups with NI files but no SoA CSV:\n")
    for m in missing_ni[:10]:  # Show first 10
        print(f"  {m['group']}: {m['ni_count']} NI files")
    if len(missing_ni) > 10:
        print(f"  ... and {len(missing_ni) - 10} more")
else:
    print("\n✓ All groups with NI files have SoA CSV files.")

print("\n" + "=" * 80)
print("READ ERRORS: SoA CSV files that couldn't be read")
print("=" * 80)
if missing_soa:
    print(f"\nFound {len(missing_soa)} SoA CSV files with read errors:\n")
    for m in missing_soa:
        print(f"  Group: {m['group']}")
        print(f"    Issue: {m['issue']}")
        print(f"    File: {m['soa_file']}")
        print()
else:
    print("\n✓ All SoA CSV files read successfully.")

# Summary statistics
total_groups_with_both = len([g for g in groups if len(g.ni_files) > 0 and g.soa_csv and g.soa_csv.exists()])
total_mismatches = len(mismatches)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total groups with both NI files and SoA CSV: {total_groups_with_both}")
print(f"Groups with matching counts: {total_groups_with_both - total_mismatches}")
print(f"Groups with mismatched counts: {total_mismatches}")
if total_groups_with_both > 0:
    match_rate = (total_groups_with_both - total_mismatches) / total_groups_with_both * 100
    print(f"Match rate: {match_rate:.1f}%")


"""
Check NI and SoA files for missing headers.

This script scans all data files and reports any files that are missing headers
or have malformed headers.
"""
import pandas as pd
from pathlib import Path
from locate_data_files import find_all_data_groups, default_base_dir

def check_ni_file_header(ni_file_path: Path):
    """Check if NI file has proper header."""
    try:
        # Try reading with header
        with open(ni_file_path, 'r') as f:
            first_line = f.readline().strip()
        
        # Check if first line looks like a header (contains column names)
        if not first_line or first_line.startswith('0') or first_line.replace('.', '').replace('-', '').replace('\t', '').replace(' ', '').isdigit():
            return False, "No header detected - first line appears to be data"
        
        # Try to parse as CSV to verify
        try:
            df = pd.read_csv(ni_file_path, sep="\t", header=0, nrows=1)
            # Check for required columns
            required_cols = ['Correct_Timing_Signal[V]', 'ACC_HIHAT[V]', 'Time[s]']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return False, f"Missing required columns: {missing_cols}"
            return True, "OK"
        except Exception as e:
            return False, f"Error parsing header: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"

def check_soa_file_header(soa_file_path: Path):
    """Check if SoA CSV file has proper header."""
    try:
        # Try reading with header
        with open(soa_file_path, 'r') as f:
            first_line = f.readline().strip()
        
        # Check if first line looks like a header
        if not first_line or first_line.replace(',', '').replace(' ', '').isdigit():
            return False, "No header detected - first line appears to be data"
        
        # Try to parse as CSV to verify
        try:
            df = pd.read_csv(soa_file_path, nrows=1)
            # Check for expected columns (may vary)
            if len(df.columns) == 0:
                return False, "No columns detected"
            return True, f"OK - columns: {list(df.columns)}"
        except Exception as e:
            return False, f"Error parsing header: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"

# Main checking
base = default_base_dir()
groups = find_all_data_groups(base)

print("Checking all NI and SoA files for header issues...\n")

ni_issues = []
soa_issues = []

for group in groups:
    # Check NI files
    for ni_file in group.ni_files:
        if ni_file.exists():
            is_ok, message = check_ni_file_header(ni_file)
            if not is_ok:
                ni_issues.append({
                    'file': str(ni_file),
                    'group': f"{group.month}/{group.subject}/{group.date}",
                    'issue': message
                })
    
    # Check SoA CSV files
    if group.soa_csv and group.soa_csv.exists():
        is_ok, message = check_soa_file_header(group.soa_csv)
        if not is_ok:
            soa_issues.append({
                'file': str(group.soa_csv),
                'group': f"{group.month}/{group.subject}/{group.date}",
                'issue': message
            })

# Report results
print("=" * 80)
print("NI FILE HEADER CHECK RESULTS")
print("=" * 80)
if ni_issues:
    print(f"\nFound {len(ni_issues)} NI files with header issues:\n")
    for issue in ni_issues:
        print(f"  File: {issue['file']}")
        print(f"  Group: {issue['group']}")
        print(f"  Issue: {issue['issue']}")
        print()
else:
    print("\n✓ All NI files have proper headers!")

print("\n" + "=" * 80)
print("SOA FILE HEADER CHECK RESULTS")
print("=" * 80)
if soa_issues:
    print(f"\nFound {len(soa_issues)} SoA CSV files with header issues:\n")
    for issue in soa_issues:
        print(f"  File: {issue['file']}")
        print(f"  Group: {issue['group']}")
        print(f"  Issue: {issue['issue']}")
        print()
else:
    print("\n✓ All SoA CSV files have proper headers!")

# Summary
total_issues = len(ni_issues) + len(soa_issues)
print("\n" + "=" * 80)
print(f"SUMMARY: Found {total_issues} files with header issues")
print("=" * 80)


"""
Find and delete extra NI files that don't have matching SoA entries.
"""
import pandas as pd
from pathlib import Path
from locate_data_files import find_all_data_groups, default_base_dir
from datetime import datetime, timedelta

def extract_time_from_ni_filename(ni_file_path):
    """Extract datetime from NI filename like '1_20250509_131208_ni.txt'"""
    try:
        parts = ni_file_path.stem.split('_')
        if len(parts) >= 3:
            date_str = parts[1]  # e.g., '20250509'
            time_str = parts[2]  # e.g., '131208'
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            hour = int(time_str[:2])
            minute = int(time_str[2:4])
            second = int(time_str[4:6])
            return datetime(year, month, day, hour, minute, second)
    except:
        pass
    return None

def count_soa_rows(soa_csv_path: Path):
    """Count rows in SoA CSV, handling missing headers."""
    try:
        soa_data_temp = pd.read_csv(soa_csv_path, nrows=1)
        expected_cols = ['name', 'Q1', 'Q2', 'datetime']
        has_header = all(col in soa_data_temp.columns for col in expected_cols)
        
        if not has_header:
            soa_data = pd.read_csv(soa_csv_path, header=None)
            if len(soa_data.columns) >= 4:
                soa_data.columns = expected_cols[:len(soa_data.columns)]
        else:
            soa_data = pd.read_csv(soa_csv_path)
        
        return soa_data
    except Exception as e:
        return None

def find_extra_ni_files(group):
    """Find NI files that don't have matching SoA entries."""
    ni_count = len(group.ni_files)
    
    if not group.soa_csv or not group.soa_csv.exists():
        return None, "No SoA CSV file"
    
    soa_data = count_soa_rows(group.soa_csv)
    if soa_data is None:
        return None, "Could not read SoA CSV"
    
    soa_row_count = len(soa_data)
    
    if ni_count <= soa_row_count:
        return None, "No extra NI files"
    
    # Convert SoA datetime
    soa_data['datetime_dt'] = pd.to_datetime(soa_data['datetime'])
    
    # Extract NI file times
    ni_files_with_times = []
    for ni_file in group.ni_files:
        ni_time = extract_time_from_ni_filename(ni_file)
        if ni_time:
            ni_files_with_times.append((ni_file, ni_time))
    
    ni_files_with_times.sort(key=lambda x: x[1])
    
    print(f"\n{'='*80}")
    print(f"Group: {group.month}/{group.subject}/{group.date}")
    print(f"NI files: {ni_count}, SoA rows: {soa_row_count}, Extra NI files: {ni_count - soa_row_count}")
    print(f"{'='*80}")
    
    # Match each SoA entry to closest NI file
    matched_ni_files = set()
    soa_matched = []
    
    for idx, soa_row in soa_data.iterrows():
        soa_time = soa_row['datetime_dt']
        
        best_ni_file = None
        best_diff = float('inf')
        
        for ni_file, ni_time in ni_files_with_times:
            if ni_file in matched_ni_files:
                continue
            
            diff = abs((soa_time - ni_time).total_seconds())
            if diff < best_diff:
                best_diff = diff
                best_ni_file = ni_file
        
        if best_ni_file:
            matched_ni_files.add(best_ni_file)
            soa_matched.append((soa_row, best_ni_file, best_diff))
    
    # Find unmatched NI files
    unmatched_ni_files = [ni_file for ni_file, _ in ni_files_with_times if ni_file not in matched_ni_files]
    
    print(f"\nMatched NI files ({len(matched_ni_files)}):")
    for soa_row, ni_file, diff in sorted(soa_matched, key=lambda x: x[1].name):
        ni_time = extract_time_from_ni_filename(ni_file)
        print(f"  {ni_file.name:35s} ↔ {soa_row['datetime_dt'].strftime('%H:%M:%S')} (diff: {diff:.0f}s)")
    
    print(f"\nUnmatched NI files ({len(unmatched_ni_files)}) - TO DELETE:")
    for ni_file in unmatched_ni_files:
        ni_time = extract_time_from_ni_filename(ni_file)
        print(f"  {ni_file.name:35s} ({ni_time.strftime('%H:%M:%S') if ni_time else 'unknown'})")
    
    return unmatched_ni_files, None

# Main execution
base = default_base_dir()
groups = find_all_data_groups(base)

# Find groups with extra NI files
problematic_groups = []
for group in groups:
    ni_count = len(group.ni_files)
    
    if group.soa_csv and group.soa_csv.exists() and ni_count > 0:
        soa_data = count_soa_rows(group.soa_csv)
        if soa_data is not None:
            soa_row_count = len(soa_data)
            
            if ni_count > soa_row_count:
                problematic_groups.append(group)

print(f"Found {len(problematic_groups)} groups with extra NI files\n")

deleted_files = []
errors = []

for group in problematic_groups:
    extra_files, error = find_extra_ni_files(group)
    
    if error:
        errors.append((group, error))
        continue
    
    if extra_files:
        print(f"\nDeleting {len(extra_files)} extra NI file(s)...")
        for ni_file in extra_files:
            try:
                if ni_file.exists():
                    file_size_mb = ni_file.stat().st_size / (1024 * 1024)
                    ni_file.unlink()
                    deleted_files.append((group, ni_file, file_size_mb))
                    print(f"  ✓ Deleted: {ni_file.name} ({file_size_mb:.1f} MB)")
                else:
                    print(f"  ⚠️  File not found: {ni_file.name}")
            except Exception as e:
                print(f"  ✗ Error deleting {ni_file.name}: {e}")
                errors.append((group, f"Error deleting {ni_file.name}: {e}"))

print(f"\n\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Groups processed: {len(problematic_groups)}")
print(f"Files deleted: {len(deleted_files)}")
print(f"Errors: {len(errors)}")

if deleted_files:
    total_size_mb = sum(size for _, _, size in deleted_files)
    print(f"\nTotal space freed: {total_size_mb:.1f} MB")
    print(f"\nDeleted files:")
    for group, ni_file, size in deleted_files:
        print(f"  {group.month}/{group.subject}/{group.date}: {ni_file.name} ({size:.1f} MB)")

if errors:
    print(f"\nErrors:")
    for group, error in errors:
        print(f"  {group.month}/{group.subject}/{group.date}: {error}")


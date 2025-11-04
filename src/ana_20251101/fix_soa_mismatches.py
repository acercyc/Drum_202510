"""
Fix SoA CSV files with mismatched row counts by identifying and removing extra rows
or flagging missing rows based on timestamp matching with NI files.
"""
import pandas as pd
from pathlib import Path
from locate_data_files import find_all_data_groups, default_base_dir
from datetime import datetime

def extract_time_from_ni_filename(ni_file_path):
    """Extract time from NI filename like '1_20250509_131208_ni.txt'"""
    try:
        parts = ni_file_path.stem.split('_')
        if len(parts) >= 3:
            time_str = parts[2]  # e.g., '131208'
            hour = int(time_str[:2])
            minute = int(time_str[2:4])
            second = int(time_str[4:6])
            return hour, minute, second
    except:
        pass
    return None, None, None

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

def fix_group_mismatch(group):
    """Fix SoA file for a group with mismatch."""
    ni_count = len(group.ni_files)
    
    if not group.soa_csv or not group.soa_csv.exists():
        return None, "No SoA CSV file"
    
    soa_data = count_soa_rows(group.soa_csv)
    if soa_data is None:
        return None, "Could not read SoA CSV"
    
    soa_row_count = len(soa_data)
    
    if ni_count == soa_row_count:
        return None, "Already matched"
    
    # Convert datetime
    soa_data['datetime_dt'] = pd.to_datetime(soa_data['datetime'])
    
    # Extract NI file times
    ni_times = []
    for ni_file in group.ni_files:
        hour, minute, second = extract_time_from_ni_filename(ni_file)
        if hour is not None:
            ni_times.append((hour, minute, second))
    
    ni_times.sort()
    
    print(f"\n{'='*80}")
    print(f"Group: {group.month}/{group.subject}/{group.date}")
    print(f"NI files: {ni_count}, SoA rows: {soa_row_count}, Difference: {ni_count - soa_row_count:+d}")
    print(f"{'='*80}")
    
    if ni_count < soa_row_count:
        # Extra SoA rows - need to remove them
        print(f"\n⚠️  Extra {soa_row_count - ni_count} SoA row(s) - will identify and remove")
        
        # Group SoA entries by time periods
        soa_data['hour'] = soa_data['datetime_dt'].dt.hour
        
        # Find the time period that matches NI files
        ni_hours = set([h for h, m, s in ni_times])
        
        # Try to match by hour first
        matching_rows = soa_data[soa_data['hour'].isin(ni_hours)].copy()
        
        if len(matching_rows) == ni_count:
            # Perfect match by hour
            rows_to_keep = matching_rows
            print(f"✓ Found {len(rows_to_keep)} rows matching NI file hours: {sorted(ni_hours)}")
        else:
            # Need more sophisticated matching - use closest time
            print(f"⚠️  Hour-based matching found {len(matching_rows)} rows, need {ni_count}")
            print(f"   Using closest time matching...")
            
            # Sort SoA by datetime
            soa_sorted = soa_data.sort_values('datetime_dt').copy()
            
            # Match each NI file to closest SoA entry
            matched_indices = []
            for ni_hour, ni_min, ni_sec in ni_times:
                ni_time = ni_hour * 3600 + ni_min * 60 + ni_sec
                
                best_idx = None
                best_diff = float('inf')
                
                for idx, row in soa_sorted.iterrows():
                    if idx in matched_indices:
                        continue
                    
                    dt = row['datetime_dt']
                    soa_time = dt.hour * 3600 + dt.minute * 60 + dt.second
                    diff = abs(soa_time - ni_time)
                    
                    if diff < best_diff:
                        best_diff = diff
                        best_idx = idx
                
                if best_idx is not None:
                    matched_indices.append(best_idx)
            
            rows_to_keep = soa_data.loc[matched_indices].copy()
            print(f"✓ Matched {len(rows_to_keep)} rows to NI files")
        
        rows_to_delete = soa_data[~soa_data.index.isin(rows_to_keep.index)].copy()
        
        print(f"\nRows to DELETE ({len(rows_to_delete)}):")
        for idx, row in rows_to_delete.iterrows():
            dt = row['datetime_dt']
            print(f"  {dt.strftime('%H:%M:%S')} - Q1={row['Q1']:3.0f}, Q2={row['Q2']:3.0f}")
        
        print(f"\nRows to KEEP ({len(rows_to_keep)}):")
        for idx, row in rows_to_keep.iterrows():
            dt = row['datetime_dt']
            print(f"  {dt.strftime('%H:%M:%S')} - Q1={row['Q1']:3.0f}, Q2={row['Q2']:3.0f}")
        
        # Save cleaned data
        rows_to_keep_clean = rows_to_keep.drop(columns=['datetime_dt', 'hour'])
        rows_to_keep_clean.to_csv(group.soa_csv, index=False)
        
        return {
            'action': 'deleted',
            'deleted': len(rows_to_delete),
            'kept': len(rows_to_keep_clean)
        }, None
        
    else:
        # Missing SoA rows - can't fix, just report
        print(f"\n⚠️  Missing {ni_count - soa_row_count} SoA row(s) - cannot fix automatically")
        print(f"   NI files exist but no corresponding SoA entries")
        print(f"\nNI file times:")
        for i, (h, m, s) in enumerate(ni_times, 1):
            print(f"  {i:2d}. {h:02d}:{m:02d}:{s:02d}")
        print(f"\nSoA entry times:")
        for i, row in soa_data.iterrows():
            dt = row['datetime_dt']
            print(f"  {i+1:2d}. {dt.strftime('%H:%M:%S')} - Q1={row['Q1']:3.0f}, Q2={row['Q2']:3.0f}")
        
        return {
            'action': 'missing',
            'missing': ni_count - soa_row_count
        }, None

# Main execution
base = default_base_dir()
groups = find_all_data_groups(base)

# Find mismatches
mismatches = []
for group in groups:
    ni_count = len(group.ni_files)
    
    if group.soa_csv and group.soa_csv.exists() and ni_count > 0:
        soa_data = count_soa_rows(group.soa_csv)
        if soa_data is not None:
            soa_row_count = len(soa_data)
            
            if ni_count != soa_row_count:
                mismatches.append(group)

print(f"Found {len(mismatches)} groups with mismatches to fix\n")

fixed = []
could_not_fix = []

for group in mismatches:
    result, error = fix_group_mismatch(group)
    if result:
        if result['action'] == 'deleted':
            fixed.append((group, result))
        else:
            could_not_fix.append((group, result))
    elif error:
        print(f"Error: {error}")

print(f"\n\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Fixed (deleted extra rows): {len(fixed)}")
for group, result in fixed:
    print(f"  ✓ {group.month}/{group.subject}/{group.date}: Deleted {result['deleted']} rows, kept {result['kept']}")

print(f"\nCould not fix (missing rows): {len(could_not_fix)}")
for group, result in could_not_fix:
    print(f"  ⚠️  {group.month}/{group.subject}/{group.date}: Missing {result['missing']} SoA row(s)")


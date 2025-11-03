"""
Demonstration script for locate_data_files.py

This script demonstrates how to use the data file locator functions to:
1. Find all data files across all participants
2. Filter by month, subject, or date
3. Access NI files and SoA files
4. Use indexing functions for organized access
"""

from locate_data_files import (
    find_all_data_groups,
    default_base_dir,
    index_by_participant,
    index_by_month,
    find_files,
    DataFileGroup
)
from pathlib import Path


def demo_basic_usage():
    """Demonstrate basic usage: finding all data groups."""
    print("=" * 80)
    print("DEMO 1: Basic Usage - Find All Data Groups")
    print("=" * 80)
    
    # Get the base directory
    base_dir = default_base_dir()
    print(f"\nBase directory: {base_dir}")
    
    # Find all data groups
    groups = find_all_data_groups(base_dir)
    print(f"\nFound {len(groups)} total data file groups")
    
    # Show first few groups
    print("\nFirst 5 groups:")
    for i, group in enumerate(groups[:5]):
        print(f"\n  Group {i+1}:")
        print(f"    Month: {group.month}")
        print(f"    Subject: {group.subject}")
        print(f"    Date: {group.date}")
        print(f"    NI files: {len(group.ni_files)}")
        print(f"    SoA CSV: {'✓' if group.soa_csv else '✗'}")
        print(f"    SoA XLSX: {'✓' if group.soa_xlsx else '✗'}")
        if group.ni_files:
            print(f"    First NI file: {group.ni_files[0].name}")


def demo_filtering():
    """Demonstrate filtering by month, subject, or date."""
    print("\n" + "=" * 80)
    print("DEMO 2: Filtering Data Groups")
    print("=" * 80)
    
    base_dir = default_base_dir()
    all_groups = find_all_data_groups(base_dir)
    
    # Filter by month
    print("\n--- Filter by Month: 5月 ---")
    may_groups = find_files(all_groups, month="5月")
    print(f"Found {len(may_groups)} groups in May")
    for group in may_groups[:3]:
        print(f"  {group.month}/{group.subject}/{group.date}: {len(group.ni_files)} NI files")
    
    # Filter by subject
    print("\n--- Filter by Subject: 3 ---")
    subject3_groups = find_files(all_groups, subject="3")
    print(f"Found {len(subject3_groups)} groups for subject 3")
    for group in subject3_groups[:3]:
        print(f"  {group.month}/{group.subject}/{group.date}: {len(group.ni_files)} NI files")
    
    # Filter by date
    print("\n--- Filter by Date: 20250507 ---")
    date_groups = find_files(all_groups, date="20250507")
    print(f"Found {len(date_groups)} groups for date 20250507")
    for group in date_groups:
        print(f"  {group.month}/{group.subject}/{group.date}: {len(group.ni_files)} NI files")
    
    # Combined filter
    print("\n--- Combined Filter: 5月, Subject 3, Date 20250507 ---")
    combined = find_files(all_groups, month="5月", subject="3", date="20250507")
    print(f"Found {len(combined)} groups")
    if combined:
        group = combined[0]
        print(f"  {group.month}/{group.subject}/{group.date}")
        print(f"  NI files ({len(group.ni_files)}):")
        for ni_file in group.ni_files:
            print(f"    - {ni_file.name}")


def demo_indexing():
    """Demonstrate indexing functions for organized access."""
    print("\n" + "=" * 80)
    print("DEMO 3: Indexing Functions")
    print("=" * 80)
    
    base_dir = default_base_dir()
    all_groups = find_all_data_groups(base_dir)
    
    # Index by participant
    print("\n--- Index by Participant ---")
    participant_index = index_by_participant(all_groups)
    print(f"Found {len(participant_index)} participants")
    
    print("\nSummary by participant:")
    for subject in sorted(participant_index.keys())[:5]:  # Show first 5
        groups = participant_index[subject]
        total_ni = sum(len(g.ni_files) for g in groups)
        dates_with_soa = sum(1 for g in groups if g.soa_csv)
        print(f"  Subject {subject}:")
        print(f"    - {len(groups)} dates")
        print(f"    - {total_ni} total NI files")
        print(f"    - {dates_with_soa} dates with SoA data")
    
    # Index by month
    print("\n--- Index by Month ---")
    month_index = index_by_month(all_groups)
    print(f"Found {len(month_index)} months")
    
    print("\nSummary by month:")
    for month in sorted(month_index.keys()):
        groups = month_index[month]
        total_ni = sum(len(g.ni_files) for g in groups)
        unique_subjects = len(set(g.subject for g in groups))
        print(f"  {month}:")
        print(f"    - {len(groups)} date/subject combinations")
        print(f"    - {unique_subjects} unique subjects")
        print(f"    - {total_ni} total NI files")


def demo_accessing_files():
    """Demonstrate how to access and work with the files."""
    print("\n" + "=" * 80)
    print("DEMO 4: Accessing Files")
    print("=" * 80)
    
    base_dir = default_base_dir()
    all_groups = find_all_data_groups(base_dir)
    
    # Find a group with both NI files and SoA data
    group_with_data = None
    for group in all_groups:
        if len(group.ni_files) > 0 and group.soa_csv:
            group_with_data = group
            break
    
    if group_with_data:
        print(f"\nExample group: {group_with_data.month}/{group_with_data.subject}/{group_with_data.date}")
        
        # Access NI files
        print(f"\nNI files ({len(group_with_data.ni_files)}):")
        for i, ni_file in enumerate(group_with_data.ni_files[:3]):  # Show first 3
            print(f"  {i+1}. {ni_file}")
            print(f"     Path: {ni_file}")
            print(f"     Exists: {ni_file.exists()}")
            if ni_file.exists():
                size_mb = ni_file.stat().st_size / (1024 * 1024)
                print(f"     Size: {size_mb:.2f} MB")
        
        # Access SoA CSV
        if group_with_data.soa_csv:
            print(f"\nSoA CSV file:")
            print(f"  Path: {group_with_data.soa_csv}")
            print(f"  Exists: {group_with_data.soa_csv.exists()}")
            
            # Show how to load it
            if group_with_data.soa_csv.exists():
                import pandas as pd
                try:
                    soa_data = pd.read_csv(group_with_data.soa_csv)
                    print(f"  Rows: {len(soa_data)}")
                    print(f"  Columns: {list(soa_data.columns)}")
                    print(f"  First few rows:")
                    print(soa_data.head(3).to_string(index=False))
                except Exception as e:
                    print(f"  Error loading: {e}")
        
        # Access SoA XLSX
        if group_with_data.soa_xlsx:
            print(f"\nSoA XLSX file:")
            print(f"  Path: {group_with_data.soa_xlsx}")
            print(f"  Exists: {group_with_data.soa_xlsx.exists()}")


def demo_processing_example():
    """Demonstrate a practical processing example."""
    print("\n" + "=" * 80)
    print("DEMO 5: Practical Processing Example")
    print("=" * 80)
    
    base_dir = default_base_dir()
    all_groups = find_all_data_groups(base_dir)
    
    # Example: Process all groups for a specific participant
    subject = "3"
    subject_groups = find_files(all_groups, subject=subject)
    
    print(f"\nProcessing all data for Subject {subject}:")
    print(f"Found {len(subject_groups)} date groups")
    
    # Count statistics
    total_ni_files = 0
    dates_with_data = 0
    
    for group in subject_groups:
        if len(group.ni_files) > 0:
            dates_with_data += 1
            total_ni_files += len(group.ni_files)
            print(f"\n  Date: {group.date}")
            print(f"    NI files: {len(group.ni_files)}")
            print(f"    SoA available: {'Yes' if group.soa_csv else 'No'}")
            
            # Example: Show first NI file name pattern
            if group.ni_files:
                first_file = group.ni_files[0]
                print(f"    Example file: {first_file.name}")
    
    print(f"\nSummary for Subject {subject}:")
    print(f"  Total dates with NI data: {dates_with_data}")
    print(f"  Total NI files: {total_ni_files}")


def demo_iterating_all_data():
    """Demonstrate how to iterate through all data systematically."""
    print("\n" + "=" * 80)
    print("DEMO 6: Iterating Through All Data")
    print("=" * 80)
    
    base_dir = default_base_dir()
    all_groups = find_all_data_groups(base_dir)
    
    # Organize by participant, then by date
    participant_index = index_by_participant(all_groups)
    
    print("\nIterating through all participants and their dates:")
    print(f"Total participants: {len(participant_index)}")
    
    for subject in sorted(participant_index.keys())[:3]:  # Show first 3 participants
        groups = participant_index[subject]
        print(f"\n  Participant {subject}:")
        print(f"    Total dates: {len(groups)}")
        
        for group in groups[:3]:  # Show first 3 dates per participant
            print(f"      Date {group.date}:")
            print(f"        NI files: {len(group.ni_files)}")
            print(f"        SoA: {'Available' if group.soa_csv else 'Not available'}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("LOCATE_DATA_FILES.PY - DEMONSTRATION")
    print("=" * 80)
    
    try:
        demo_basic_usage()
        demo_filtering()
        demo_indexing()
        demo_accessing_files()
        demo_processing_example()
        demo_iterating_all_data()
        
        print("\n" + "=" * 80)
        print("ALL DEMONSTRATIONS COMPLETE")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()






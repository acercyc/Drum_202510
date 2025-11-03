"""
Utility module to locate and organize NI sensor data files and SoA rating files
from all participants, months, and session dates.

Data structure: data/{Month}/{Subject}/NI/{YYYYMMDD}/ and data/{Month}/{Subject}/SoA/{YYYYMMDD}/
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class DataFileGroup:
    """Container for all data files for a specific month/subject/date combination."""
    month: str           # e.g., "4月"
    subject: str         # e.g., "2"
    date: str            # e.g., "20250409"
    ni_files: List[Path] # All NI files for this date (sorted by run number)
    soa_csv: Optional[Path]
    soa_xlsx: Optional[Path]


def find_date_directories(parent: Path) -> List[Path]:
    """
    Find all date directories (YYYYMMDD format) in a parent directory.
    
    Args:
        parent: Parent directory to search
        
    Returns:
        Sorted list of date directory paths
    """
    if not parent.exists():
        return []
    return sorted([p for p in parent.iterdir() 
                   if p.is_dir() and p.name.isdigit() and len(p.name) == 8], 
                  key=lambda p: p.name)


def get_month_subject_from_path(path: Path) -> Tuple[str, str]:
    """
    Extract month and subject from a path.
    
    Expected structure: .../{Month}/{Subject}/...
    
    Args:
        path: Path to extract from
        
    Returns:
        Tuple of (month, subject)
    """
    parts = path.parts
    # Find subject (digit) and month (before subject)
    for i, part in enumerate(parts):
        if part.isdigit() and i > 0:
            subject = part
            month = parts[i-1]
            return month, subject
    return "unknown", "unknown"


def find_all_data_groups(base_dir: Path) -> List[DataFileGroup]:
    """
    Scan the data directory structure and find all data file groups.
    
    Scans data/{Month}/{Subject}/NI/{Date}/ and data/{Month}/{Subject}/SoA/{Date}/
    to find all combinations of month/subject/date with their associated files.
    
    Args:
        base_dir: Base directory containing the data folder structure
                  (e.g., project_root/data)
        
    Returns:
        List of DataFileGroup objects, one per month/subject/date combination
    """
    groups: List[DataFileGroup] = []
    
    # Iterate through month directories (4月, 5月, etc.)
    for month_dir in base_dir.iterdir():
        if not month_dir.is_dir():
            continue
            
        month = month_dir.name
        
        # Iterate through subject directories (1, 2, 3, etc.)
        for subject_dir in month_dir.iterdir():
            if not subject_dir.is_dir() or not subject_dir.name.isdigit():
                continue
                
            subject = subject_dir.name
            
            # Check for NI and SoA directories
            ni_dir = subject_dir / "NI"
            soa_dir = subject_dir / "SoA"
            
            # Find all date directories in NI folder
            ni_dates = find_date_directories(ni_dir) if ni_dir.exists() else []
            
            # Find all date directories in SoA folder
            soa_dates = find_date_directories(soa_dir) if soa_dir.exists() else []
            
            # Get all unique dates (union of NI and SoA dates)
            all_dates = sorted(set([d.name for d in ni_dates] + [d.name for d in soa_dates]))
            
            # Create DataFileGroup for each date
            for date in all_dates:
                # Find NI files for this date
                ni_files = []
                ni_date_dir = ni_dir / date if ni_dir.exists() else None
                if ni_date_dir and ni_date_dir.exists():
                    ni_files = sorted(ni_date_dir.glob("*_ni.txt"), 
                                    key=lambda p: int(p.stem.split('_')[0]) if p.stem.split('_')[0].isdigit() else 0)
                
                # Find SoA files for this date
                soa_csv = None
                soa_xlsx = None
                soa_date_dir = soa_dir / date if soa_dir.exists() else None
                if soa_date_dir and soa_date_dir.exists():
                    csv_candidates = sorted(soa_date_dir.glob("*.csv"))
                    xlsx_candidates = sorted(soa_date_dir.glob("*.xlsx"))
                    soa_csv = csv_candidates[0] if csv_candidates else None
                    soa_xlsx = xlsx_candidates[0] if xlsx_candidates else None
                
                # Only create group if there are NI files or SoA files
                if ni_files or soa_csv or soa_xlsx:
                    groups.append(
                        DataFileGroup(
                            month=month,
                            subject=subject,
                            date=date,
                            ni_files=ni_files,
                            soa_csv=soa_csv,
                            soa_xlsx=soa_xlsx
                        )
                    )
    
    return groups


def index_by_participant(groups: List[DataFileGroup]) -> Dict[str, List[DataFileGroup]]:
    """
    Group DataFileGroup objects by subject (participant) number.
    
    Args:
        groups: List of DataFileGroup objects
        
    Returns:
        Dictionary mapping subject number to list of DataFileGroup objects
    """
    indexed: Dict[str, List[DataFileGroup]] = {}
    for group in groups:
        if group.subject not in indexed:
            indexed[group.subject] = []
        indexed[group.subject].append(group)
    
    # Sort each participant's groups by date
    for subject in indexed:
        indexed[subject].sort(key=lambda g: g.date)
    
    return indexed


def index_by_month(groups: List[DataFileGroup]) -> Dict[str, List[DataFileGroup]]:
    """
    Group DataFileGroup objects by month.
    
    Args:
        groups: List of DataFileGroup objects
        
    Returns:
        Dictionary mapping month to list of DataFileGroup objects
    """
    indexed: Dict[str, List[DataFileGroup]] = {}
    for group in groups:
        if group.month not in indexed:
            indexed[group.month] = []
        indexed[group.month].append(group)
    
    # Sort each month's groups by subject and date
    for month in indexed:
        indexed[month].sort(key=lambda g: (g.subject, g.date))
    
    return indexed


def find_files(groups: List[DataFileGroup], 
               month: Optional[str] = None, 
               subject: Optional[str] = None, 
               date: Optional[str] = None) -> List[DataFileGroup]:
    """
    Query function to find DataFileGroup objects matching specific criteria.
    
    Args:
        groups: List of DataFileGroup objects to filter
        month: Filter by month (e.g., "4月"). If None, no filtering.
        subject: Filter by subject number (e.g., "2"). If None, no filtering.
        date: Filter by date (e.g., "20250409"). If None, no filtering.
        
    Returns:
        Filtered list of DataFileGroup objects matching the criteria
    """
    filtered = groups
    
    if month is not None:
        filtered = [g for g in filtered if g.month == month]
    
    if subject is not None:
        filtered = [g for g in filtered if g.subject == subject]
    
    if date is not None:
        filtered = [g for g in filtered if g.date == date]
    
    return filtered


def default_base_dir() -> Path:
    """Get the default base directory for the data structure."""
    # project_root = .../Drum
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "data"


def main() -> None:
    """Command-line interface for listing data file groups."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="List all data file groups (NI and SoA) from the dataset"
    )
    parser.add_argument(
        "--base",
        type=str,
        default=str(default_base_dir()),
        help="Base directory containing the data folder (default: project/data)",
    )
    parser.add_argument(
        "--month",
        type=str,
        help="Filter by month (e.g., 4月)",
    )
    parser.add_argument(
        "--subject",
        type=str,
        help="Filter by subject number (e.g., 2)",
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Filter by date (e.g., 20250409)",
    )
    args = parser.parse_args()

    base_path = Path(args.base)
    groups = find_all_data_groups(base_path)
    
    # Apply filters if specified
    if args.month or args.subject or args.date:
        groups = find_files(groups, month=args.month, subject=args.subject, date=args.date)
    
    # Print summary
    print(f"Found {len(groups)} data file groups")
    print("\n" + "="*80)
    print(f"{'Month':<8} {'Subject':<10} {'Date':<12} {'NI Files':<12} {'SoA CSV':<20}")
    print("="*80)
    
    for group in groups:
        ni_count = len(group.ni_files)
        soa_status = "✓" if group.soa_csv else "✗"
        print(f"{group.month:<8} {group.subject:<10} {group.date:<12} {ni_count:<12} {soa_status:<20}")
    
    # Show detailed info for first few groups
    if groups:
        print("\n" + "="*80)
        print("Sample groups (first 3):")
        print("="*80)
        for group in groups[:3]:
            print(f"\n{group.month}/{group.subject}/{group.date}:")
            print(f"  NI files ({len(group.ni_files)}):")
            for ni_file in group.ni_files[:5]:  # Show first 5
                print(f"    - {ni_file.name}")
            if len(group.ni_files) > 5:
                print(f"    ... and {len(group.ni_files) - 5} more")
            if group.soa_csv:
                print(f"  SoA CSV: {group.soa_csv.name}")
            if group.soa_xlsx:
                print(f"  SoA XLSX: {group.soa_xlsx.name}")


if __name__ == "__main__":
    main()






import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class SoAFiles:
    date: str
    csv_path: Optional[str]
    xlsx_path: Optional[str]


@dataclass
class NISoAPair:
    date: str
    ni_file: str
    soa_csv: Optional[str]
    soa_xlsx: Optional[str]


@dataclass
class SoAToNIs:
    date: str
    soa_csv: Optional[str]
    soa_xlsx: Optional[str]
    ni_files: List[str]


def find_date_directories(parent: Path) -> List[Path]:
    if not parent.exists():
        return []
    return sorted([p for p in parent.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: p.name)


def index_ni_files(ni_root: Path) -> Dict[str, List[Path]]:
    date_to_files: Dict[str, List[Path]] = {}
    for date_dir in find_date_directories(ni_root):
        ni_files = sorted(date_dir.glob("*_ni.txt"))
        if ni_files:
            date_to_files[date_dir.name] = ni_files
    return date_to_files


def index_soa_files(soa_root: Path) -> Dict[str, SoAFiles]:
    date_to_soa: Dict[str, SoAFiles] = {}
    for date_dir in find_date_directories(soa_root):
        csv_candidates = sorted(date_dir.glob("*.csv"))
        xlsx_candidates = sorted(date_dir.glob("*.xlsx"))
        csv_path = str(csv_candidates[0]) if csv_candidates else None
        xlsx_path = str(xlsx_candidates[0]) if xlsx_candidates else None
        date_to_soa[date_dir.name] = SoAFiles(date=date_dir.name, csv_path=csv_path, xlsx_path=xlsx_path)
    return date_to_soa


def build_pairs(ni_root: Path, soa_root: Path) -> List[NISoAPair]:
    date_to_ni = index_ni_files(ni_root)
    date_to_soa = index_soa_files(soa_root)
    shared_dates = sorted(set(date_to_ni.keys()) & set(date_to_soa.keys()))

    pairs: List[NISoAPair] = []
    for date in shared_dates:
        soa = date_to_soa.get(date)
        for ni_file in date_to_ni.get(date, []):
            pairs.append(
                NISoAPair(
                    date=date,
                    ni_file=str(ni_file),
                    soa_csv=soa.csv_path if soa else None,
                    soa_xlsx=soa.xlsx_path if soa else None,
                )
            )
    return pairs


def build_grouped(ni_root: Path, soa_root: Path) -> List[SoAToNIs]:
    """Group NI files under each SoA date (one SoA -> many NI)."""
    date_to_ni = index_ni_files(ni_root)
    date_to_soa = index_soa_files(soa_root)
    shared_dates = sorted(set(date_to_ni.keys()) & set(date_to_soa.keys()))

    grouped: List[SoAToNIs] = []
    for date in shared_dates:
        soa = date_to_soa.get(date)
        ni_files = [str(p) for p in date_to_ni.get(date, [])]
        grouped.append(
            SoAToNIs(
                date=date,
                soa_csv=soa.csv_path if soa else None,
                soa_xlsx=soa.xlsx_path if soa else None,
                ni_files=ni_files,
            )
        )
    return grouped


def default_base_dir() -> Path:
    # project_root = .../Drum
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "data" / "raw_20250911"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List paired NI and SoA data files by date"
    )
    parser.add_argument(
        "--base",
        type=str,
        default=str(default_base_dir()),
        help="Base directory containing NI/ and SoA/ (default: project/data/raw_20250911)",
    )
    parser.add_argument(
        "--format",
        choices=["tsv", "json"],
        default="tsv",
        help="Output format (default: tsv)",
    )
    parser.add_argument(
        "--mode",
        choices=["grouped", "pairs"],
        default="grouped",
        help="Output mode: grouped (one SoA -> many NI) or pairs (one row per NI)",
    )
    args = parser.parse_args()

    base_path = Path(args.base)
    ni_root = base_path / "NI"
    soa_root = base_path / "SoA"

    if args.mode == "grouped":
        grouped = build_grouped(ni_root, soa_root)
        if args.format == "json":
            print(json.dumps([asdict(g) for g in grouped], ensure_ascii=False, indent=2))
        else:
            # TSV: one row per date (SoA), NI files joined by ';'
            print("date\tsoa_csv\tsoa_xlsx\tni_files_count\tni_files")
            for g in grouped:
                joined = ";".join(g.ni_files)
                print(f"{g.date}\t{g.soa_csv or ''}\t{g.soa_xlsx or ''}\t{len(g.ni_files)}\t{joined}")
    else:
        pairs = build_pairs(ni_root, soa_root)
        if args.format == "json":
            print(json.dumps([asdict(p) for p in pairs], ensure_ascii=False, indent=2))
        else:
            print("date\tni_file\tsoa_csv\tsoa_xlsx")
            for p in pairs:
                print(f"{p.date}\t{p.ni_file}\t{p.soa_csv or ''}\t{p.soa_xlsx or ''}")


if __name__ == "__main__":
    main()

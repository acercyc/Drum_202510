# Drum Dataset

## Overview

This dataset contains drum performance data collected over multiple months in 2025. The data is organized by month and subject, with various data modalities including MIDI sequences, position tracking, video recordings, and analysis spreadsheets.

**Dataset Statistics:**
- **Total Size:** ~179 GB
- **Total Files:** 1,938 files
  - 1,662 text files (.txt)
  - 86 Excel files (.xlsx)
  - 64 video files (.mov)
  - Additional JSON files (position data)
- **Time Period:** April 2025 - October 2025
- **Total Subjects:** 14 subjects across 6 months

## Directory Structure

```
data/
├── 4月/          (April - 2 subjects)
├── 5月/          (May - 2 subjects)
├── 6月/          (June - 2 subjects)
├── 7月/          (July - 2 subjects)
├── 9月/          (September - 3 subjects)
└── 10月/         (October - 3 subjects)
```

## Subject Directory Structure

Each subject directory (numbered: 1, 2, 3, etc.) contains the following subdirectories:

```
{Month}/{SubjectNumber}/
├── MIDI/              # MIDI sequence data
├── pos/               # Position tracking data
├── movie/             # Video recordings
├── SoA/               # State of Art analysis spreadsheets
├── memo/              # Memos/notes (optional)
├── NI/                # Additional data (optional)
└── ADio/              # Audio data (optional)
```

## Data Types

### 1. MIDI Data (`MIDI/`)

**Structure:**
```
MIDI/
└── {YYYYMMDD}/        # Date directories (e.g., 20250409)
    ├── midi_1回目.txt  # First attempt
    ├── midi_2回目.txt  # Second attempt
    ├── midi_3回目.txt  # Third attempt
    └── ...
```

**File Format:**
- **Format:** CSV text files
- **Content:** Comma-separated values, appears to contain note and timestamp pairs
- **Example:** `42,3.581560` (note number, timestamp)
- **Naming:** Files are numbered sequentially with Japanese counter (回目 = "times/attempts")

**Characteristics:**
- Multiple recording sessions per date
- Typically 1-8+ attempts per session
- Files range from ~1.6KB to ~3KB each

### 2. Position Data (`pos/`)

**Structure:**
```
pos/
└── {YYYYMMDD}/                    # Date directories
    ├── pos_data_list_1回目.json   # Position data for first attempt
    ├── pos_data_list_2回目.json   # Position data for second attempt
    └── ...
```

**File Format:**
- **Format:** JSON files
- **Size:** ~4.9MB - 6.7MB per file
- **Content:** Position tracking data corresponding to MIDI attempts
- **Naming:** Matches MIDI file numbering scheme

### 3. Video Recordings (`movie/`)

**Structure:**
```
movie/
└── {YYYYMMDD}/                    # Date directories
    └── {YYYY-MM-DD HH-MM-SS}.mov  # Timestamped video files
```

**File Format:**
- **Format:** QuickTime movie files (.mov)
- **Naming:** Files use timestamp format: `YYYY-MM-DD HH-MM-SS.mov`
- **Example:** `2025-04-09 13-33-57.mov`

### 4. SoA Analysis (`SoA/`)

**Structure:**
```
SoA/
├── {SubjectNumber}_SoA.xlsx       # Root level summary (some subjects)
└── {YYYYMMDD}/                    # Date directories
    └── {YYYYMMDD}_SoA.xlsx        # Date-specific analysis
```

**File Format:**
- **Format:** Microsoft Excel files (.xlsx)
- **Content:** State of Art analysis spreadsheets
- **Naming:** Date-based naming convention

### 5. Optional Directories

- **`memo/`**: Memos and notes (not present in all subjects)
- **`NI/`**: Additional data (not present in all subjects)
- **`ADio/`**: Audio data (not present in all subjects)

## Data Organization by Month

| Month | Japanese Name | Subjects | Dates Range |
|-------|--------------|----------|-------------|
| April | 4月 | 2 | Subject 1, 2 |
| May | 5月 | 2 | Subject 3, 4 |
| June | 6月 | 2 | Subject 5, 6 |
| July | 7月 | 2 | Subject 7, 8 |
| September | 9月 | 3 | Subject 9, 10, 11 |
| October | 10月 | 3 | Subject 12, 13, 14 |

## Example Data Path

```
data/4月/2/MIDI/20250409/midi_1回目.txt
data/4月/2/pos/20250409/pos_data_list_1回目.json
data/4月/2/movie/20250409/2025-04-09 13-33-57.mov
data/4月/2/SoA/20250409/20250409_SoA.xlsx
```

## Notes

- The dataset uses Japanese month names (4月, 5月, etc.) for month directories
- File naming conventions use Japanese counters (回目) for attempt numbering
- Some subjects may have incomplete data (missing certain subdirectories)
- Date formats are consistent: YYYYMMDD for directories, YYYY-MM-DD HH-MM-SS for video files
- The dataset appears to be from a longitudinal study tracking drum performance over time

## Access

The `data/` directory in this repository is a symbolic link pointing to:
```
/mnt/DataDrive/Share with Acer/20251002_drum_dataset_actual test/drive-download-20251002T061429Z-1/
```


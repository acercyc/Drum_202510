# Transfer Entropy Analysis for New Data Structure (2025-11-01)

## Overview

This analysis script adapts the Transfer Entropy (TE) computation from `ana_TE_20250911.py` to work with the new data structure organized by months and subjects.

**Note:** The main sensor data files are located in the **NI/** folder, which contains high-frequency (10 kHz) sensor data including acceleration sensors, EMG signals, and the reference timing signal. The MIDI folder contains alternative MIDI sequence data.

## Main Analysis Goal

The project's main analysis is to **compute Transfer Entropy from the reference signal (`Correct_Timing_Signal[V]`) to the Hi-hat signal (`ACC_HIHAT[V]`) in NI files**.

- **Source Signal**: `Correct_Timing_Signal[V]` - Reference timing signal (model/right hand timing)
- **Target Signal**: `ACC_HIHAT[V]` - Participant's hi-hat performance timing
- **Transfer Entropy**: Measures the information flow from the reference timing to the participant's actual performance

This analysis quantifies how well the participant's performance aligns with the reference timing pattern, providing an objective measure of synchronization and timing accuracy.

## Data Structure

The new data structure is:
```
data/
├── {Month}/          (e.g., 4月, 5月, 6月)
│   └── {Subject}/    (e.g., 1, 2, 3)
│       ├── NI/                    # IMPORTANT: Main sensor data files
│       │   └── {YYYYMMDD}/
│       │       └── {Run}_{Date}_{Time}_ni.txt
│       └── SoA/                   # Sense of Agency ratings
│           └── {YYYYMMDD}/
│               ├── tomita_soa_*.csv
│               └── {YYYYMMDD}_SoA.xlsx
```

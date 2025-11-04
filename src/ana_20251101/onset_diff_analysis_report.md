# Onset Difference Analysis Report

**Analysis Date:** 2025-11-01  
**Script:** `ana_onset_diff_20251101.py`  
**Output File:** `onset_diff_results_all.csv`

---

## Overview

This analysis computes timing differences (action errors) between the reference timing signal and participants' hi-hat performance in drumming sessions. The onset difference serves as a quantitative measure of timing accuracy, where smaller absolute differences indicate better synchronization with the reference timing.

### Main Analysis Goal

- **Reference Signal:** `Correct_Timing_Signal[V]` - Reference timing signal (model/right hand timing)
- **Target Signal:** `ACC_HIHAT[V]` - Participant's hi-hat performance timing
- **Onset Difference:** Measures timing error between participant's hits and reference timing

---

## Methodology

### 1. Data Sources

- **NI Files:** Tab-separated CSV files containing sensor data from drum performance sessions
- **SoA Files:** CSV files containing Sense of Agency (SoA) ratings (Q1: pre-SoA, Q2: post-SoA)
- **Data Filtering:** Only sessions with both NI files and SoA data were included in the analysis

### 2. Signal Processing

#### Onset Detection
- **Algorithm:** Threshold-based onset detection
- **Parameters:**
  - `threshold = 0.1` - Signal threshold for detecting onsets
  - `minimal_interval = 1000` samples - Minimum samples between detected onsets
- **Processing:** Raw signals are used directly (no convolution smoothing applied)

#### Onset Matching
- For each target onset (participant's hi-hat hit), find the nearest reference onset
- Compute time difference: `time_difference = target_onset_time - reference_onset_time`
- Positive values indicate the participant hit after the reference; negative values indicate hitting before

### 3. Computed Metrics

For each session, the following statistics are computed:

- **`n_ref_onsets`:** Number of onsets detected in reference signal
- **`n_target_onsets`:** Number of onsets detected in target signal
- **`n_matched_onsets`:** Number of successfully matched onsets
- **`mean_diff`:** Mean timing difference (signed, in seconds)
- **`std_diff`:** Standard deviation of timing differences
- **`mean_abs_diff`:** Mean absolute timing difference (magnitude of error)
- **`median_diff`:** Median timing difference
- **`median_abs_diff`:** Median absolute timing difference
- **`min_diff`:** Minimum timing difference
- **`max_diff`:** Maximum timing difference

### 4. Data Integration

- Onset difference results are matched with SoA ratings (Q1, Q2) based on session order
- Results are grouped by subject, date, and month
- Missing data handling: Sessions with failed onset detection are included with NaN values (later removed)

---

## Results Summary

### Overall Statistics

- **Total Sessions Analyzed:** 527
- **Subjects:** 10 (subjects 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
- **Total Dates:** 30 unique dates
- **Total Matched Onsets:** 132,092

### Timing Difference Statistics

- **Mean Onset Difference:** -0.0564 ± 0.2342 seconds
  - Slight negative bias indicates participants tend to hit slightly before the reference timing on average
- **Mean Absolute Onset Difference:** 0.1853 ± 0.2525 seconds
  - Average timing error magnitude
- **Median Absolute Onset Difference:** 0.0736 seconds
  - Median timing error magnitude (more robust to outliers)

### Matched Onsets

- **Mean Matched Onsets per Session:** 250.6 ± 118.9
- Range varies across subjects and sessions

### Subject Breakdown

| Subject | Sessions | Mean Abs Diff (s) | Std Abs Diff (s) | Mean Matched Onsets |
|---------|----------|-------------------|------------------|---------------------|
| 3       | 78       | 0.2301           | 0.2872           | 267.8               |
| 4       | 49       | 0.1713           | 0.2051           | 226.8               |
| 5       | 73       | 0.2096           | 0.3071           | 194.1               |
| 6       | 54       | 0.2375           | 0.3584           | 169.2               |
| 7       | 55       | 0.1816           | 0.1904           | 227.6               |
| 8       | 34       | 0.2487           | 0.2788           | 196.3               |
| 9       | 81       | 0.1187           | 0.1602           | 308.0               |
| 10      | 52       | 0.1624           | 0.1898           | 327.2               |
| 11      | 45       | 0.1139           | 0.1869           | 314.0               |
| 12      | 6        | 0.2613           | 0.3055           | 250.3               |

**Observations:**
- Subjects 9 and 11 show the lowest mean absolute differences (best timing accuracy)
- Subject 12 has the highest mean absolute difference but only 6 sessions
- Subjects 9, 10, and 11 have the highest number of matched onsets per session

---

## Correlations with Sense of Agency (SoA) Ratings

### Mean Onset Difference (Signed)

- **MeanDiff vs Q1 (pre-SoA):** r = 0.2090 (n=526)
- **MeanDiff vs Q2 (post-SoA):** r = 0.2072 (n=526)

**Interpretation:** Small positive correlations suggest that higher (more positive) timing differences (hitting later) are weakly associated with higher SoA ratings.

### Mean Absolute Onset Difference (Magnitude)

- **MeanAbsDiff vs Q1 (pre-SoA):** r = -0.2221 (n=526)
- **MeanAbsDiff vs Q2 (post-SoA):** r = -0.2180 (n=526)

**Interpretation:** Small negative correlations suggest that larger timing errors (regardless of direction) are weakly associated with lower SoA ratings. This indicates that better timing accuracy (smaller errors) is associated with higher sense of agency.

---

## Data Quality and Processing Notes

### Data Cleaning

1. **Missing Headers:** Some SoA CSV files were missing headers. The script automatically detects and handles this by:
   - Checking for expected column names (`name`, `Q1`, `Q2`, `datetime`)
   - Reading without header if missing and assigning standard column names

2. **Missing Data Removal:** 
   - 3 sessions had missing onset difference data (failed onset detection)
   - These rows were removed from the final dataset
   - Final dataset: 527 complete sessions

3. **Data Consistency:**
   - Ensured matching counts between NI files and SoA rows
   - Fixed inconsistencies by removing extra SoA rows or extra NI files
   - Final consistency: 100% match rate for all groups with both data types

### Session Filtering

- Only sessions with both NI files and SoA data were included
- Excluded incomplete groups (missing either NI files or SoA data)
- This ensures all analyzed sessions have corresponding subjective ratings

---

## Visualization

### Plotting Script: `plot_onset_diff_by_subject.py`

Creates time-series plots for each subject showing:
- **Mean Onset Difference** (green triangles) - Action error over time
- **Q1 Ratings** (blue circles) - Pre-SoA ratings
- **Q2 Ratings** (orange squares) - Post-SoA ratings
- **Date Separation** - Vertical lines separate different dates
- **Correlations** - Displayed in top-right corner

**Usage:**
```bash
# Plot all subjects
python plot_onset_diff_by_subject.py

# Plot specific subject
python plot_onset_diff_by_subject.py --subject 4

# Show plots interactively
python plot_onset_diff_by_subject.py --show
```

**Output:** Plots saved to `plots_onset_diff/` directory (10 plots, one per subject)

---

## Key Findings

1. **Timing Accuracy:**
   - Median absolute timing error: ~74 ms (0.0736 seconds)
   - Mean absolute timing error: ~185 ms (0.1853 seconds)
   - Participants show slight tendency to hit slightly early (negative mean difference)

2. **Subject Variability:**
   - Wide range of timing accuracy across subjects
   - Subjects 9 and 11 show best timing accuracy (lowest mean absolute differences)
   - Subject variability suggests individual differences in motor timing abilities

3. **SoA Relationship:**
   - Small but consistent negative correlation between timing error magnitude and SoA ratings
   - Better timing accuracy (smaller errors) associated with higher sense of agency
   - Suggests that temporal precision contributes to sense of agency

4. **Temporal Evolution:**
   - Onset differences can be tracked over time across multiple sessions
   - Allows investigation of learning effects and practice-related improvements
   - Individual plots show subject-specific patterns and trends

---

## Technical Details

### Key Functions

1. **`onset_detection()`**
   - Threshold-based onset detection algorithm
   - Parameters: threshold (0.1), minimal_interval (1000 samples)
   - Returns: onset array and onset indices

2. **`find_nearest_onset_time()`**
   - Matches each target onset to nearest reference onset
   - Computes time differences
   - Returns: nearest reference times and time differences

3. **`compute_onset_diff_from_ni_file()`**
   - Main computation function for a single NI file
   - Extracts signals, detects onsets, matches them, computes statistics
   - Returns: dictionary with all computed metrics

### Data Structure

**Input Files:**
- NI files: Tab-separated CSV with columns including `Time[s]`, `Correct_Timing_Signal[V]`, `ACC_HIHAT[V]`
- SoA files: CSV with columns `name`, `Q1`, `Q2`, `datetime`

**Output File:** `onset_diff_results_all.csv`
- Columns: `month`, `subject`, `date`, `ni_file`, `n_ref_onsets`, `n_target_onsets`, `n_matched_onsets`, `mean_diff`, `std_diff`, `mean_abs_diff`, `median_diff`, `median_abs_diff`, `min_diff`, `max_diff`, `name`, `Q1`, `Q2`, `datetime`

---

## Future Directions

1. **Temporal Analysis:** Investigate changes in timing accuracy over time within and across sessions
2. **Error Patterns:** Analyze distributions of timing errors (early vs. late, consistency)
3. **Comparison with TE:** Compare onset difference patterns with Transfer Entropy results
4. **Learning Effects:** Examine improvement trajectories for individual subjects
5. **Context Effects:** Investigate how timing accuracy varies by task conditions or session characteristics

---

## Files Generated

- **`onset_diff_results_all.csv`:** Complete results dataset (527 rows)
- **`plot_onset_diff_by_subject.py`:** Visualization script
- **`plots_onset_diff/`:** Directory containing 10 subject-specific plots
  - `onset_diff_SoA_subject_3.png` through `onset_diff_SoA_subject_12.png`

---

## References

- Analysis script: `src/ana_20251101/ana_onset_diff_20251101.py`
- Visualization script: `src/ana_20251101/plot_onset_diff_by_subject.py`
- Related analysis: Transfer Entropy analysis (`ana_TE_20251101.py`)

---

*Report generated: 2025-11-01*


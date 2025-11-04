# %%
"""
Onset Difference Analysis for New Data Structure (2025-11-01)

This script analyzes drum performance data by computing timing differences (action errors)
between the reference timing signal (Correct_Timing_Signal[V]) and the participant's 
hi-hat performance (ACC_HIHAT[V]) in NI files.

Main Analysis Goal:
- Reference Signal: Correct_Timing_Signal[V] - Reference timing signal (model/right hand timing)
- Target Signal: ACC_HIHAT[V] - Participant's hi-hat performance timing
- Onset Difference: Measures timing error between participant's hits and reference timing
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from scipy.signal import convolve

# %%
# Set working directory to the file directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# %%
# Import local utilities
from locate_data_files import find_all_data_groups, default_base_dir, index_by_participant

# %%
# Convolution functions for signal smoothing
def impulse_response_convolution(
    data, peak_amplitude=1.0, decay_rate=0.001, response_length=200
):
    """
    Applies an impulse response function with exponential decay using convolution.

    Parameters:
        data (numpy array): The input time series (e.g., binary impulses).
        peak_amplitude (float): The peak height of the response.
        decay_rate (float): The exponential decay rate (higher means faster decay).
        response_length (int): The length of the impulse response function.

    Returns:
        numpy array: The output signal with impulse responses applied.
    """
    # Generate the impulse response kernel
    data = np.abs(data)
    t = np.arange(response_length)
    impulse_response_kernel = peak_amplitude * np.exp(-decay_rate * t)

    # Apply convolution
    output_signal = convolve(data, impulse_response_kernel, mode="full")[: len(data)]

    # Normalize the output signal
    if np.max(output_signal) > 0:
        output_signal /= np.max(output_signal)

    return output_signal


# %%
def onset_detection(time_series: np.ndarray, 
                    threshold: float = 0.1, 
                    minimal_interval: int = 1000) -> tuple:
    """
    Detect onsets in a time series signal.
    
    Args:
        time_series: Input time series array
        threshold: Threshold for detecting onsets
        minimal_interval: Minimum samples between onsets
        
    Returns:
        Tuple of (onsets_array, onset_indices)
    """
    onsets = np.zeros_like(time_series)
    onset_indices = []
    
    in_onset = False
    last_onset_idx = -minimal_interval
    
    for i in range(len(time_series)):
        if time_series[i] > threshold and not in_onset and (i - last_onset_idx >= minimal_interval):
            onsets[i] = 1.0
            onset_indices.append(i)
            last_onset_idx = i
            in_onset = True
        elif time_series[i] <= threshold and in_onset:
            in_onset = False
    
    return onsets, np.array(onset_indices)


# %%
def find_nearest_onset_time(target_onset_time, reference_onset_time):
    """
    Find the nearest onset time in the reference onset time for each target onset time.
    Return the nearest onset time and the corresponding time difference.
    
    Parameters:
        target_onset_time (np.array): Array with the target onset times (in seconds).
        reference_onset_time (np.array): Array with the reference onset times (in seconds).
    
    Returns:
        np.array: Array with the nearest reference onset times.
        np.array: Array with the time differences (target - reference, in seconds).
    """
    nearest_onset_time = []
    time_difference = []
    
    for target_time in target_onset_time:
        nearest_time = reference_onset_time[np.argmin(np.abs(reference_onset_time - target_time))]
        nearest_onset_time.append(nearest_time)
        time_difference.append(target_time - nearest_time)
    
    return np.array(nearest_onset_time), np.array(time_difference)


# %%
def compute_onset_diff_from_ni_file(ni_file_path: Path,
                                    threshold: float = 0.1,
                                    minimal_interval: int = 1000):
    """
    Compute onset differences (action errors) from an NI file.
    
    Extracts Correct_Timing_Signal[V] (reference) and ACC_HIHAT[V] (target),
    processes them, detects onsets, and computes timing differences.
    
    Args:
        ni_file_path: Path to NI file (tab-separated CSV)
        threshold: Threshold for onset detection (default: 0.1)
        minimal_interval: Minimum samples between onsets (default: 1000)
        
    Returns:
        Dictionary with onset difference statistics, or None if computation fails
    """
    try:
        # Load NI data
        data = pd.read_csv(ni_file_path, sep="\t", header=0)
        
        # Extract signals
        if 'Correct_Timing_Signal[V]' not in data.columns:
            print(f"Warning: Correct_Timing_Signal[V] not found in {ni_file_path}")
            return None
        if 'ACC_HIHAT[V]' not in data.columns:
            print(f"Warning: ACC_HIHAT[V] not found in {ni_file_path}")
            return None
        
        reference = data['Correct_Timing_Signal[V]'].values
        target = data['ACC_HIHAT[V]'].values
        t = data['Time[s]'].values
        
        # Detect onsets in both signals (using raw signals, no convolution)
        _, reference_onset_idx = onset_detection(reference, threshold=threshold, minimal_interval=minimal_interval)
        _, target_onset_idx = onset_detection(target, threshold=threshold, minimal_interval=minimal_interval)
        
        if len(reference_onset_idx) == 0:
            print(f"Warning: No onsets detected in reference signal for {ni_file_path.name}")
            return None
        if len(target_onset_idx) == 0:
            print(f"Warning: No onsets detected in target signal for {ni_file_path.name}")
            return None
        
        # Convert onset indices to time values
        reference_onset_times = t[reference_onset_idx]
        target_onset_times = t[target_onset_idx]
        
        # Find nearest reference onset for each target onset and compute differences
        nearest_ref_times, time_differences = find_nearest_onset_time(target_onset_times, reference_onset_times)
        
        if len(time_differences) == 0:
            print(f"Warning: No valid onset differences computed for {ni_file_path.name}")
            return None
        
        # Compute summary statistics
        mean_diff = np.mean(time_differences)
        std_diff = np.std(time_differences)
        mean_abs_diff = np.mean(np.abs(time_differences))
        median_diff = np.median(time_differences)
        median_abs_diff = np.median(np.abs(time_differences))
        min_diff = np.min(time_differences)
        max_diff = np.max(time_differences)
        
        return {
            'n_reference_onsets': len(reference_onset_times),
            'n_target_onsets': len(target_onset_times),
            'n_matched_onsets': len(time_differences),
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'mean_abs_diff': mean_abs_diff,
            'median_diff': median_diff,
            'median_abs_diff': median_abs_diff,
            'min_diff': min_diff,
            'max_diff': max_diff,
            'reference_onset_times': reference_onset_times,
            'target_onset_times': target_onset_times,
            'time_differences': time_differences
        }
        
    except Exception as e:
        print(f"Error computing onset differences for {ni_file_path}: {e}")
        return None


# %%
# Find all data files
base = default_base_dir()
groups = find_all_data_groups(base)
print(f"Found {len(groups)} data file groups")

# Show summary
print("\nSummary by participant:")
participant_index = index_by_participant(groups)
for subject, subject_groups in sorted(participant_index.items()):
    total_ni_files = sum(len(g.ni_files) for g in subject_groups)
    dates_with_soa = sum(1 for g in subject_groups if g.soa_csv)
    print(f"  Subject {subject}: {len(subject_groups)} dates, {total_ni_files} NI files, {dates_with_soa} dates with SoA")

# %%
# Filter groups: Only include sessions with BOTH NI files AND SoA data
print("\nFiltering groups: Only including sessions with both NI files and SoA data...")
complete_groups = []
for group in groups:
    has_ni = len(group.ni_files) > 0
    has_soa = (group.soa_csv is not None and group.soa_csv.exists())
    
    if has_ni and has_soa:
        complete_groups.append(group)
    else:
        missing = []
        if not has_ni:
            missing.append("NI files")
        if not has_soa:
            missing.append("SoA data")
        print(f"  Skipping {group.month}/{group.subject}/{group.date}: Missing {', '.join(missing)}")

print(f"\nFound {len(complete_groups)} complete groups (out of {len(groups)} total)")
print(f"Excluded {len(groups) - len(complete_groups)} incomplete groups")

# %%
# Main processing loop
print("\nStarting onset difference computation...")
data_all = []

for i, group in enumerate(complete_groups):
    
    print(f"\nProcessing group {i+1}/{len(complete_groups)}: {group.month}/{group.subject}/{group.date}")
    print(f"  {len(group.ni_files)} NI files")
    
    # Load the SoA data (should always exist due to filtering, but check anyway)
    soa_data = None
    if group.soa_csv and group.soa_csv.exists():
        try:
            # Try reading with header first
            soa_data_temp = pd.read_csv(group.soa_csv, nrows=1)
            
            # Check if header is missing (columns don't match expected names)
            expected_cols = ['name', 'Q1', 'Q2', 'datetime']
            has_header = all(col in soa_data_temp.columns for col in expected_cols)
            
            if not has_header:
                # File is missing header - read without header and add column names
                soa_data = pd.read_csv(group.soa_csv, header=None)
                if len(soa_data.columns) >= 4:
                    soa_data.columns = expected_cols[:len(soa_data.columns)]
                    print(f"  Warning: SoA CSV missing header, added column names")
                else:
                    print(f"  Warning: SoA CSV has unexpected number of columns: {len(soa_data.columns)}")
            else:
                # File has proper header - read normally
                soa_data = pd.read_csv(group.soa_csv)
            
            print(f"  Loaded SoA data: {len(soa_data)} rows")
        except Exception as e:
            print(f"  Error: Could not load SoA CSV: {e}")
            print(f"  Skipping this group due to SoA loading error")
            continue
    else:
        print(f"  Error: SoA CSV not found (should not happen after filtering)")
        continue
    
    # Compute onset differences for all NI files in this group
    onset_results = []
    ni_file_names = []
    
    for ni_file in group.ni_files:
        print(f"    Computing onset differences for {ni_file.name}...")
        result = compute_onset_diff_from_ni_file(ni_file)
        
        if result is not None:
            onset_results.append(result)
            ni_file_names.append(ni_file.name)
            print(f"      Found {result['n_matched_onsets']} matched onsets")
            print(f"      Mean diff: {result['mean_diff']:.4f}s, Mean abs diff: {result['mean_abs_diff']:.4f}s")
        else:
            # Still add entry with NaN values
            onset_results.append({
                'n_reference_onsets': np.nan,
                'n_target_onsets': np.nan,
                'n_matched_onsets': np.nan,
                'mean_diff': np.nan,
                'std_diff': np.nan,
                'mean_abs_diff': np.nan,
                'median_diff': np.nan,
                'median_abs_diff': np.nan,
                'min_diff': np.nan,
                'max_diff': np.nan
            })
            ni_file_names.append(ni_file.name)
    
    # Create result DataFrame for this group
    if len(onset_results) > 0:
        group_results = pd.DataFrame({
            'month': [group.month] * len(onset_results),
            'subject': [group.subject] * len(onset_results),
            'date': [group.date] * len(onset_results),
            'ni_file': ni_file_names,
            'n_ref_onsets': [r['n_reference_onsets'] for r in onset_results],
            'n_target_onsets': [r['n_target_onsets'] for r in onset_results],
            'n_matched_onsets': [r['n_matched_onsets'] for r in onset_results],
            'mean_diff': [r['mean_diff'] for r in onset_results],
            'std_diff': [r['std_diff'] for r in onset_results],
            'mean_abs_diff': [r['mean_abs_diff'] for r in onset_results],
            'median_diff': [r['median_diff'] for r in onset_results],
            'median_abs_diff': [r['median_abs_diff'] for r in onset_results],
            'min_diff': [r['min_diff'] for r in onset_results],
            'max_diff': [r['max_diff'] for r in onset_results]
        })
        
        # Add SoA data if available
        if soa_data is not None:
            # Match onset results to SoA rows
            # Simple approach: take first N SoA rows, or pad with NaN
            n_soa = len(soa_data)
            n_onset = len(onset_results)
            
            if n_onset <= n_soa:
                # Use first N SoA rows
                soa_to_merge = soa_data.iloc[:n_onset].reset_index(drop=True)
            else:
                # Pad SoA data with NaN rows
                nan_rows = pd.DataFrame({col: [np.nan] * (n_onset - n_soa) for col in soa_data.columns})
                soa_to_merge = pd.concat([soa_data, nan_rows], ignore_index=True)
            
            # Merge SoA data with group results
            group_results = group_results.join(soa_to_merge, how='left')
        
        data_all.append(group_results)
        print(f"  Added {len(group_results)} rows to results")

# %%
# Combine all results
if len(data_all) > 0:
    data_all_combined = pd.concat(data_all, ignore_index=True)
    print(f"\nTotal rows in combined data: {len(data_all_combined)}")
    
    # Save results
    output_file = "onset_diff_results_all.csv"
    data_all_combined.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    # Display summary statistics
    print("\nSummary statistics:")
    if 'subject' in data_all_combined.columns:
        summary = data_all_combined.groupby('subject').agg({
            'mean_abs_diff': ['count', 'mean', 'std'],
            'n_matched_onsets': 'mean'
        })
        print(summary)
else:
    print("\nWarning: No data was processed!")

# %%


# %%
"""
Transfer Entropy Analysis for New Data Structure (2025-11-01)

This script analyzes drum performance data using Transfer Entropy (TE) to measure
information flow from the reference timing signal (Correct_Timing_Signal[V]) to the
participant's hi-hat performance (ACC_HIHAT[V]) in NI files.

Main Analysis Goal:
- Source Signal: Correct_Timing_Signal[V] - Reference timing signal (model/right hand timing)
- Target Signal: ACC_HIHAT[V] - Participant's hi-hat performance timing
- Transfer Entropy: Measures information flow from reference timing to participant's performance
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from scipy.signal import convolve

# %%
# Set working directory to the file directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# %%
# Add parent directory to path for TE module
sys.path.append("../")

from TE import TransferEntropyCalculator_continuous_kraskov

# %%
# Import local utilities
from locate_data_files import find_all_data_groups, default_base_dir, index_by_participant, find_files

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
def compute_TE_from_ni_file(ni_file_path: Path,
                            sampling_rate: int = 10000,
                            down_sampling_rate: int = 100):
    """
    Compute Transfer Entropy from an NI file.
    
    Extracts Correct_Timing_Signal[V] (source) and ACC_HIHAT[V] (target),
    processes them, and computes Transfer Entropy.
    
    Args:
        ni_file_path: Path to NI file (tab-separated CSV)
        sampling_rate: Original sampling rate in Hz (default: 10000)
        down_sampling_rate: Target sampling rate after downsampling (default: 100 Hz)
        
    Returns:
        Transfer Entropy value (float), or np.nan if computation fails
    """
    try:
        # Load NI data
        data = pd.read_csv(ni_file_path, sep="\t", header=0)
        
        # Extract signals
        if 'Correct_Timing_Signal[V]' not in data.columns:
            print(f"Warning: Correct_Timing_Signal[V] not found in {ni_file_path}")
            return np.nan
        if 'ACC_HIHAT[V]' not in data.columns:
            print(f"Warning: ACC_HIHAT[V] not found in {ni_file_path}")
            return np.nan
        
        source = data['Correct_Timing_Signal[V]'].values
        target = data['ACC_HIHAT[V]'].values
        t = data['Time[s]'].values
        
        # Apply convolution smoothing
        decay_rate = 0.001
        response_length = 1000
        target_cov = impulse_response_convolution(
            target, peak_amplitude=1, decay_rate=decay_rate, response_length=response_length
        )
        source_cov = impulse_response_convolution(
            source, peak_amplitude=1, decay_rate=decay_rate, response_length=response_length
        )
        
        # Find signal boundaries (non-zero regions)
        onsets_array, onset_idx = onset_detection(source_cov, threshold=0.1, minimal_interval=1000)
        if len(onset_idx) == 0:
            print(f"Warning: No onsets detected in source signal for {ni_file_path.name}")
            return np.nan
        
        idx_start = onset_idx[0]
        idx_end = onset_idx[-1]
        
        # Truncate the signals
        target_ = target_cov[idx_start:idx_end]
        source_ = source_cov[idx_start:idx_end]
        
        # Down sample the signals
        down_sample_point = int(sampling_rate / down_sampling_rate)
        target_ = target_[::down_sample_point]
        source_ = source_[::down_sample_point]
        
        # Ensure minimum length for TE computation
        if len(target_) < 100 or len(source_) < 100:
            print(f"Warning: Signal too short for TE computation: {len(target_)} samples")
            return np.nan
        
        # Compute Transfer Entropy
        te_calc = TransferEntropyCalculator_continuous_kraskov()
        te_result = te_calc.compute_TE(source_, target_, isPrintEstimation=False)
        
        return te_result
        
    except Exception as e:
        print(f"Error computing TE for {ni_file_path}: {e}")
        return np.nan


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
print("\nStarting TE computation...")
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
    
    # Compute TE for all NI files in this group
    te_results = []
    ni_file_names = []
    
    for ni_file in group.ni_files:
        print(f"    Computing TE for {ni_file.name}...")
        te_result = compute_TE_from_ni_file(ni_file)
        te_results.append(te_result)
        ni_file_names.append(ni_file.name)
        if not np.isnan(te_result):
            print(f"      TE = {te_result:.4f}")
    
    # Create result DataFrame for this group
    if len(te_results) > 0:
        group_results = pd.DataFrame({
            'month': [group.month] * len(te_results),
            'subject': [group.subject] * len(te_results),
            'date': [group.date] * len(te_results),
            'ni_file': ni_file_names,
            'TE': te_results
        })
        
        # Add SoA data if available
        if soa_data is not None:
            # Match TE results to SoA rows
            # Simple approach: take first N SoA rows, or pad with NaN
            n_soa = len(soa_data)
            n_te = len(te_results)
            
            if n_te <= n_soa:
                # Use first N SoA rows
                soa_to_merge = soa_data.iloc[:n_te].reset_index(drop=True)
            else:
                # Pad SoA data with NaN rows
                nan_rows = pd.DataFrame({col: [np.nan] * (n_te - n_soa) for col in soa_data.columns})
                soa_to_merge = pd.concat([soa_data, nan_rows], ignore_index=True)
            
            # Merge SoA data with group results
            # Use join to avoid column conflicts
            group_results = group_results.join(soa_to_merge, how='left')
        
        data_all.append(group_results)
        print(f"  Added {len(group_results)} rows to results")

# %%
# Combine all results
if len(data_all) > 0:
    data_all_combined = pd.concat(data_all, ignore_index=True)
    print(f"\nTotal rows in combined data: {len(data_all_combined)}")
    
    # Save results
    output_file = "TE_results_all.csv"
    data_all_combined.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    # Display summary statistics
    print("\nSummary statistics:")
    if 'subject' in data_all_combined.columns:
        summary = data_all_combined.groupby('subject')['TE'].agg(['count', 'mean', 'std'])
        print(summary)
else:
    print("\nWarning: No data was processed!")

# %%
# Plot the TE results
if len(data_all) > 0 and 'TE' in data_all_combined.columns:
    plt.figure(figsize=(12, 6))
    
    # Plot TE over time/index
    valid_te = data_all_combined['TE'].dropna()
    if len(valid_te) > 0:
        plt.plot(valid_te.index, valid_te.values, 
                label='TE', marker='o', linestyle='-', markersize=4, alpha=0.6)
        plt.xlabel('Index')
        plt.ylabel('Transfer Entropy')
        plt.title('Transfer Entropy Results for All Participants')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("TE_results_plot.png", dpi=150)
        print("\nPlot saved to: TE_results_plot.png")
        plt.show()
    else:
        print("No valid TE values to plot")

# %%
# Combined plot with Q1, Q2, and TE (if available)
# if len(data_all) > 0 and all(col in data_all_combined.columns for col in ['Q1', 'Q2', 'TE']):
#     import matplotlib.dates as mdates
    
#     fig, ax1 = plt.subplots(figsize=(12, 6))
#     ax2 = ax1.twinx()
    
#     # Filter to rows with valid TE
#     plot_data = data_all_combined.dropna(subset=['TE'])
    
#     if len(plot_data) > 0:
#         x_data = plot_data.index
        
#         lns1 = ax1.plot(x_data, plot_data['Q1'], label="Q1", 
#                        color="tab:blue", marker="o", markersize=4, alpha=0.7)
#         lns2 = ax1.plot(x_data, plot_data['Q2'], label="Q2", 
#                        color="tab:orange", marker="o", markersize=4, alpha=0.7)
#         lns3 = ax2.plot(x_data, plot_data['TE'], label="TE", 
#                        color="tab:green", marker="o", linestyle="--", markersize=4, alpha=0.7)
        
#         ax1.set_xlabel("Index")
#         ax1.set_ylabel("Q1 & Q2")
#         ax2.set_ylabel("TE")
        
#         lines = lns1 + lns2 + lns3
#         labels = [l.get_label() for l in lines]
#         ax1.legend(lines, labels, loc="upper left")
#         ax1.grid(True, alpha=0.3)
        
#         plt.title('SoA Ratings (Q1, Q2) and Transfer Entropy')
#         plt.tight_layout()
#         plt.savefig("TE_SoA_combined_plot.png", dpi=150)
#         print("Combined plot saved to: TE_SoA_combined_plot.png")
#         plt.show()

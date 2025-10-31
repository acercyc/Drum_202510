# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

raw_data_path = "../../data/raw_20241030/output_txt/"
participant_ids = ["datahayase1030", "datahosokawa1030", "datasawada1030", "datasugata1030"]
trial_ids = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

# %%
def import_data(file_path):
    """
    Import data from a file and add appropriate column names.

    Parameters:
    - file_path (str): Path to the data file.

    Returns:
    - pd.DataFrame: DataFrame with the data and appropriate column names.
    """
    # Define column names based on the context
    column_names = [
        "Time", 
        "Hi-hat", 
        "Snare drum", 
        "Hi-hat (cue)", 
        "Snare (cue)", 
        "EMG (right 1)", 
        "EMG (right 2)", 
        "EMG (left 1)", 
        "EMG (left 2)"
    ]

    # Load the dataset with the defined column names
    data = pd.read_csv(file_path, delim_whitespace=True, header=None, names=column_names)
    return data

# %%
def onset_detection(timeSeries, threshold=0.5, minimal_interval=100):
    """
    Detect the onset of a signal based on a threshold.

    Parameters:
    - timeSeries Time series data.
    - threshold (float): Threshold value for the onset detection.
    - minimal_interval (int): Minimal interval between two onsets.

    Returns:
    - pd.Series: Time series data with onset detection.
    """
    # Initialize the onset detection
    onset = False
    onset_time = 0
    onsets = []
    
    # absolute value of the time series
    timeSeries = np.abs(timeSeries)
    
    # normalize the time series between 0 and 1
    timeSeries = (timeSeries - timeSeries.min()) / (timeSeries.max() - timeSeries.min())

    # Iterate through the time series data
    for i in range(len(timeSeries)):
        # Check if the current value is above the threshold
        if timeSeries[i] > threshold:
            # Check if there is no onset detected or the minimal interval is passed
            if not onset or i - onset_time > minimal_interval:
                onset = True
                onset_time = i
                onsets.append(1)
            else:
                onsets.append(0)
        else:
            onsets.append(0)
            
    # conver to index of the time series use numpy
    onsets = np.array(onsets)
    onset_idx = np.where(onsets == 1)[0]
    
    return onsets, onset_idx

# %%
def extract_onset_time(data, coumn_name, threshold=0.1, minimal_interval=1000):
    """
    Extract the onset time from the data.

    Parameters:
    - data (pd.DataFrame): Data with the time series.
    - column_name (str): Column name of the time series.
    - threshold (float): Threshold value for the onset detection.
    - minimal_interval (int): Minimal interval between two onsets.

    Returns:
    - np.array: Array with the onset times.
    """
    onsets, onset_idx = onset_detection(data[coumn_name], threshold=threshold, minimal_interval=minimal_interval)
    onset_time = data['Time'][onset_idx].values
    return onset_time

# %%
def find_nearest_onset_time(target_onset_time, reference_onset_time):
    """
    Find the nearest onset time in the reference onset time for each target onset time. Return the nearest onset time and the corresponding time difference.

    Parameters:
    - target_onset_time (np.array): Array with the target onset times.
    - reference_onset_time (np.array): Array with the reference onset times.

    Returns:
    - np.array: Array with the nearest onset times.
    - np.array: Array with the time differences.
    """
    nearest_onset_time = []
    time_difference = []

    for target_time in target_onset_time:
        nearest_time = reference_onset_time[np.argmin(np.abs(reference_onset_time - target_time))]
        nearest_onset_time.append(nearest_time)
        time_difference.append(target_time - nearest_time)

    return np.array(nearest_onset_time), np.array(time_difference)

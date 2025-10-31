# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

raw_data_path = (
    "../../data/raw_20250225/コホート予備実験/cleaned_folder/cleaned_data_folder/"
)

participant_ids = [
    "20250221",
    "20250226",
    "20250228",
    "20250305",
    "20250307",
    "20250312",
]


def construct_file_path(participant_id, trial_id):
    file_path = os.path.join(
        raw_data_path, participant_id, f"cleaned_data_list_{trial_id}回目.txt"
    )
    return file_path

def find_trials(participant_id):
    trials = []
    for trial_id in range(1, 20):
        file_path = construct_file_path(participant_id, trial_id)
        if os.path.exists(file_path):
            trials.append(trial_id)
    return trials


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
        "EMG (left 2)",
    ]

    # Load the dataset with the defined column names
    data = pd.read_csv(file_path, delim_whitespace=True, header=0, names=column_names, usecols=range(9))
    return data


def import_emg_data(participant_id, trial_id):
    # data\raw_20250225\コホート予備実験\cleaned_folder\cleaned_data_folder\EMG_Data_20250221.csv
    file_path = construct_file_path(participant_id, trial_id)
    data = import_data(file_path)
    return data


def import_soa_rating_data(participant_id):
    # data\raw_20250225\コホート予備実験\cleaned_folder\cleaned_data_folder\SoA_Ratings_20250221.csv
    file_path = os.path.join(raw_data_path, f"SoA_Ratings_{participant_id}.csv")
    data = pd.read_csv(file_path)
    return data



# %%


def onset_detection(timeSeries, threshold=0.1, minimal_interval=1000):
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
    onsets_array = np.array(onsets)
    onset_idx = np.where(onsets_array == 1)[0]

    return onsets_array, onset_idx


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
    onsets_array, onset_idx = onset_detection(
        data[coumn_name], threshold=threshold, minimal_interval=minimal_interval
    )
    onset_time = data["Time"][onset_idx].values
    return onset_time
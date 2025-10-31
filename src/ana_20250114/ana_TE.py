# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

def detect_event_timings(data, cue_column, ignore_interval=0):
    """
    Detect the event onsets for a given cue column, optionally ignoring events
    that occur within a specified interval of a previous event.

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing the cue column.
    - cue_column (str): Name of the cue column to detect onsets from.
    - ignore_interval (float): Time interval to ignore subsequent events after an onset (default is 0).

    Returns:
    - list: List of detected onset times.
    """
    cue_indices = data[data[cue_column].diff() > 0].index
    detected_onsets = [data.loc[cue_indices[0], "Time"]] if len(cue_indices) > 0 else []

    for idx in cue_indices[1:]:
        if data.loc[idx, "Time"] - detected_onsets[-1] > ignore_interval:
            detected_onsets.append(data.loc[idx, "Time"])

    return detected_onsets

# Function to reconstruct the cue columns to only have "1" at event onsets
def reconstruct_cue_columns(data, cue_column, event_timings):
    """
    Reconstruct the cue column such that only the event onset has a value of 1, 
    and all other entries are set to 0.
    
    Parameters:
    - data (pd.DataFrame): Original dataset.
    - cue_column (str): Name of the cue column to reconstruct.
    - event_timings (list): List of event timings for the cue column.
    
    Returns:
    - pd.Series: Reconstructed cue column.
    """
    reconstructed_column = np.zeros(len(data))
    for timing in event_timings:
        closest_index = (data["Time"] - timing).abs().idxmin()
        reconstructed_column[closest_index] = 1
    return pd.Series(reconstructed_column, name=cue_column)


def convolve_with_exponential_kernel(data, column, decay_rate):
    """
    Convolve a column with an exponential kernel for smoothing.
    
    Parameters:
    - data (pd.DataFrame): The dataset containing the column.
    - column (str): The name of the column to convolve.
    - decay_rate (float): The decay rate (lambda) for the exponential kernel.
    
    Returns:
    - pd.Series: The convolved column.
    """
    kernel_size = int(1 / decay_rate * 100)  # Ensure kernel size is reasonable
    if kernel_size < 1:
        kernel_size = 1  # Minimum size to avoid empty kernel
    kernel = np.exp(-decay_rate * np.linspace(0, 1, kernel_size))
    kernel /= kernel.sum()  # Normalize kernel to maintain scale

    smoothed_column = np.convolve(data[column], kernel, mode='same')
    return pd.Series(smoothed_column, name=column)


file_path = "D:/SynologyDrive/Drive-Acer/DeepWen/deepwen/home/acercyc/Projects/Drum/data/raw_20241030/output_txt/datahayase1030_1.txt"
data = import_data(file_path)
# Detecting event timings for both Hi-hat and Snare cue columns
hihat_event_timings = detect_event_timings(data, "Hi-hat (cue)", ignore_interval=0.3)
snare_event_timings = detect_event_timings(data, "Snare (cue)", ignore_interval=0.3)

# Creating a dataframe for the event timings
event_timings_df = pd.DataFrame({
    "Hi-hat Event Timings": pd.Series(hihat_event_timings),
    "Snare Event Timings": pd.Series(snare_event_timings)
})

# Reconstructing the Hi-hat and Snare cue columns
data["Hi-hat (cue)"] = reconstruct_cue_columns(data, "Hi-hat (cue)", hihat_event_timings)
data["Snare (cue)"] = reconstruct_cue_columns(data, "Snare (cue)", snare_event_timings)


# Applying the exponential convolution to the Hi-hat and Snare cue columns
decay_rate = 0.1  # Adjusted decay rate for practical kernel size
data["Hi-hat (cue)"] = convolve_with_exponential_kernel(data, "Hi-hat (cue)", decay_rate)
data["Snare (cue)"] = convolve_with_exponential_kernel(data, "Snare (cue)", decay_rate)



# %%
# Plotting the smoothed cue columns in two subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Hi-hat (cue) subplot
axes[0].plot(data["Time"], data["Hi-hat (cue)"], label="Hi-hat (cue)", color='blue', alpha=0.8)
axes[0].set_title("Hi-hat (cue)")
axes[0].set_ylabel("Cue Intensity")
axes[0].grid(alpha=0.3)
axes[0].legend()

# Snare (cue) subplot
axes[1].plot(data["Time"], data["Snare (cue)"], label="Snare (cue)", color='green', alpha=0.8)
axes[1].set_title("Snare (cue)")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Cue Intensity")
axes[1].grid(alpha=0.3)
axes[1].legend()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()


columns_to_plot = data.columns[1:]  # Exclude the "Time" column
# Plotting all columns in subplots
num_columns = len(columns_to_plot)
fig, axes = plt.subplots(num_columns, 1, figsize=(15, 3 * num_columns), sharex=True)

for i, column in enumerate(columns_to_plot):
    axes[i].plot(data["Time"], data[column], label=column, alpha=0.8)
    axes[i].set_title(column)
    axes[i].set_ylabel("Value")
    axes[i].grid(alpha=0.3)
    axes[i].legend()

# Adding a common x-axis label
plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()


# %% compute transfer entropy from cue to EMG
import numpy as np
# include parent directory in the path
import sys
sys.path.append("D:/SynologyDrive/Drive-Acer/DeepWen/deepwen/home/acercyc/Projects/Drum/src")
from TE import TransferEntropyCalculator_continuous

# Initialize the TE calculator class
te_calculator = TransferEntropyCalculator_continuous(
    normalise=True, 
    kernel_width=1
)

# downsample the data to 100 Hz
data_ = data.iloc[::10, :]
data_.head()

# Compute transfer entropy from the Hi-hat cue to the EMG (right 1) column
source_array = data_["Hi-hat (cue)"].values
dest_array = data_["EMG (right 1)"].values
te_value_hihat_to_emg = te_calculator.compute_TE(source_array, dest_array)
print(f"Transfer Entropy (Hi-hat -> EMG (right 1)): {te_value_hihat_to_emg:.4f} nats")


# impliment the moving window TE computation


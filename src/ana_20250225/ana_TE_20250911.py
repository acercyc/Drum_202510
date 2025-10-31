# %%
import os
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from jpype import *
import tqdm
import matplotlib.pyplot as plt
from scipy.signal import convolve



# %%
from utils import participant_ids, raw_data_path, find_trials, import_soa_rating_data, import_emg_data, onset_detection, extract_onset_time

# %%
# set working directory to the file directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


#%%
# locate jitd jar file
from pathlib import Path
import sys

sys.path.append(
    "../"
)


from TE import TransferEntropyCalculator_continuous_kraskov

# %%
# Find files in folders and subfolders
from pair_data_paths import build_grouped, default_base_dir
base = default_base_dir()
grouped = build_grouped(base / "NI", base / "SoA")  # list of SoAToNIs
for g in grouped:
    print(g.date, g.soa_csv, len(g.ni_files))



# %%
# convolue
def convolve_with_exponential_kernel(data, decay_rate, kernel_size):
    """
    Convolve a column with an exponential kernel for smoothing.

    Parameters:
    - data (np.array): The data to be smoothed.
    - decay_rate (float): The decay rate (lambda) for the exponential kernel.

    Returns:
    - np.array: The convolved data.
    """
    kernel = np.exp(-decay_rate * np.linspace(0, 1, kernel_size))
    # kernel /= kernel.sum()  # Normalize kernel to maintain scale

    data_cov = np.convolve(data, kernel, mode="same")
    return data_cov


def impulse_response_convolution(
    data, peak_amplitude=1.0, decay_rate=0.001, response_length=200
):
    """
    Applies an impulse response function with exponential decay using convolution.

    Parameters:
        time_series (numpy array): The input time series (e.g., binary impulses).
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
    output_signal /= np.max(output_signal)

    return output_signal


def compute_TE_from_input_file(file_path):
    # Load data
    data = pd.read_csv(file_path)
    data = pd.read_csv(file_path, sep="\t")

    
    # Extract signals
    target = data['ACC_HIHAT[V]'].values
    source = data['Correct_Timing_Signal[V]'].values
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
    onsets_array, onset_idx = onset_detection(source, threshold=0.1, minimal_interval=1000)
    idx_start = onset_idx[0]
    idx_end = onset_idx[-1]
    
    # Truncate the signals
    target_ = target_cov[idx_start:idx_end]
    source_ = source_cov[idx_start:idx_end]
    t_ = t[idx_start:idx_end]
    
    # Down sample the signals
    original_sampling_rate = 10000
    down_sampling_rate = 100  # Hz
    down_sample_point = int(original_sampling_rate / down_sampling_rate)
    target_ = target_[::down_sample_point]
    source_ = source_[::down_sample_point]
    t_ = t_[::down_sample_point]
    
    # Compute Transfer Entropy
    te_calc = TransferEntropyCalculator_continuous_kraskov()
    te_result = te_calc.compute_TE(source_, target_, isPrintEstimation=True)
    
    return te_result


# %% 
data_all = []
for g in grouped:
    # load the soa data
    soa_data = pd.read_csv(g.soa_csv)
    print(soa_data.head())
    te_results = []

    for ni_file in g.ni_files:
        # compute the TE
        print(ni_file)
        te_result = compute_TE_from_input_file(ni_file)
        print(f"Transfer Entropy result: {te_result}")

        # save the TE result
        te_results.append(te_result)
    
    soa_data['TE'] = te_results
    data_all.append(soa_data)



# %%
data_all = pd.concat(data_all) 
data_all.to_csv("TE_results_all.csv", index=False)


# %%
# plot the TE results
plt.figure(figsize=(10, 5))
plt.plot(data_all['datetime'], data_all['TE'], label='TE')
plt.title('TE Results for All Participants')
plt.xlabel('Time')
plt.ylabel('TE')
plt.legend()
plt.grid(True)
plt.show()

# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# assume data_all is a DataFrame with columns: name, Q1, Q2, datetime, TE
df = data_all.copy()
# df["datetime"] = pd.to_datetime(df["datetime"])
# df = df.sort_values("datetime")

fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()

lns1 = ax1.plot(df["datetime"], df["Q1"], label="Q1", color="tab:blue", marker="o")
lns2 = ax1.plot(df["datetime"], df["Q2"], label="Q2", color="tab:orange", marker="o")
lns3 = ax2.plot(df["datetime"], df["TE"], label="TE", color="tab:green", marker="o", linestyle="--")

ax1.set_xlabel("Time")
ax1.set_ylabel("Q1 & Q2")
ax2.set_ylabel("TE")

lines = lns1 + lns2 + lns3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper left")
ax1.grid(True, alpha=0.3)

# Robust datetime tick formatting and rotation
ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
fig.autofmt_xdate(rotation=90)

plt.tight_layout()
plt.show()
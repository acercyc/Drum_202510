# %% 
import pandas as pd
import os
import re
import numpy as np
from matplotlib import pyplot as plt


filepath_raw = 'raw/'

# %%
def parse_filenames_in_folder(data_folder):
    """
    This function parses the filenames in the specified folder, extracting participant ID, date, condition, and trial number,
    and prints a table of the results.
    """
    # Define a regular expression to capture the participant ID, date, condition, and trial number
    filename_pattern = r'(\w+)_(\d{6})_(LEDvib|LED)(\d+)\.txt'

    # Create a list to store the parsed information
    file_info_list = []

    # Loop through all files in the data folder
    for filename in os.listdir(data_folder):
        if filename.endswith('.txt'):            
            # Match the filename to the pattern
            match = re.match(filename_pattern, filename)
            if match:
                participant_id, date, condition, trial_number = match.groups()
                trial_number = int(trial_number)

                # Store the parsed information
                file_info = {
                    'filename': filename,
                    'participant_id': participant_id,
                    'date': date,
                    'condition': condition,
                    'trial_number': trial_number
                }
                file_info_list.append(file_info)

    # Convert the file information to a DataFrame for easier analysis
    file_info_df = pd.DataFrame(file_info_list)
    return file_info_df


# Example usage:
# Just call the function and update the folder path inside it
parsed_df = parse_filenames_in_folder(filepath_raw)

# Display the parsed DataFrame
print(parsed_df)


# %%

# Define the file path or name
# file_path = 'raw/J_240718_LEDvib1.txt'
file_path = 'raw/A_240627_LEDvib5.txt.txt'

# Load the data using pandas read_csv
data = pd.read_csv(file_path, 
                   delim_whitespace=True,    # This tells pandas that the delimiter is whitespace
                   header=0)                 # Use the first line as the header

# if the file name contain "vib"
if 'vib' in file_path:
    # change the column names as Time[s]  HiHat[V]  RightHand[V]  Snare[V]  LeftHand[V]
    data.columns = ['Time[s]', 'HiHat[V]', 'RightHand[V]', 'Snare[V]', 'LeftHand[V]']
else:
    # change the column names as Time[s] HiHat[V] Hand[V]
    data.columns = ['Time[s]', 'HiHat[V]', 'Hand[V]']
    
        
print(data.head())

# %%
# save the data to a csv file with the same name
if not os.path.exists('data'):
    os.makedirs('data')
data.to_csv('data/' + file_path.split('/')[1].split('.')[0] + '.csv', index=False)

# %%
# Defining a function for onset detection for both "HiHat" and "Hand" signals
def detect_onsets(data, hihat_threshold=1, hand_threshold=0.1, ignore_duration_hihat=0.3, ignore_duration_hand=0.2):
    # Onset detection for "HiHat"
    hihat_onsets = []
    vibrating_hihat = False
    last_onset_time_hihat = -ignore_duration_hihat
    
    for i in range(1, len(data)):
        current_time = data['Time[s]'][i]
        if data['HiHat[V]'][i] > hihat_threshold and not vibrating_hihat and (current_time - last_onset_time_hihat >= ignore_duration_hihat):
            hihat_onsets.append(current_time)
            last_onset_time_hihat = current_time
            vibrating_hihat = True
        elif data['HiHat[V]'][i] <= hihat_threshold and vibrating_hihat:
            vibrating_hihat = False
    
    # Onset detection for "Hand"
    hand_onsets = []
    vibrating_hand = False
    last_onset_time_hand = -ignore_duration_hand
    
    for i in range(1, len(data)):
        current_time = data['Time[s]'][i]
        if data['Hand[V]'][i] > hand_threshold and not vibrating_hand and (current_time - last_onset_time_hand >= ignore_duration_hand):
            hand_onsets.append(current_time)
            last_onset_time_hand = current_time
            vibrating_hand = True
        elif data['Hand[V]'][i] <= hand_threshold and vibrating_hand:
            vibrating_hand = False
    
    return hihat_onsets, hand_onsets

# Testing the function on the dataset
hihat_onsets_result, hand_onsets_result = detect_onsets(data)

# Displaying the number of detected onsets for both HiHat and Hand
len(hihat_onsets_result), len(hand_onsets_result), hihat_onsets_result[:5], hand_onsets_result[:5]  # Displaying the first 5 onsets of each

# %%
# Defining a function to plot the time series and onsets for both HiHat and Hand signals
def plot_onsets(data, hihat_onsets, hand_onsets, zoom_start=None, zoom_end=None):
    plt.figure(figsize=(12, 6))
    
    # If zoom_start and zoom_end are provided, filter the data
    if zoom_start and zoom_end:
        filtered_data = data[(data['Time[s]'] >= zoom_start) & (data['Time[s]'] <= zoom_end)]
        filtered_hihat_onsets = [onset for onset in hihat_onsets if zoom_start <= onset <= zoom_end]
        filtered_hand_onsets = [onset for onset in hand_onsets if zoom_start <= onset <= zoom_end]
    else:
        filtered_data = data
        filtered_hihat_onsets = hihat_onsets
        filtered_hand_onsets = hand_onsets
    
    # Plotting HiHat signal
    plt.plot(filtered_data['Time[s]'], filtered_data['HiHat[V]'], label='HiHat[V]', color='blue')
    
    # Mark the HiHat onsets
    for onset in filtered_hihat_onsets:
        plt.axvline(x=onset, color='red', linestyle='--', alpha=0.6, label='HiHat Onset' if onset == filtered_hihat_onsets[0] else "")
    
    # Plotting Hand signal
    plt.plot(filtered_data['Time[s]'], filtered_data['Hand[V]'], label='Hand[V]', color='orange')
    
    # Mark the Hand onsets
    for onset in filtered_hand_onsets:
        plt.axvline(x=onset, color='green', linestyle='--', alpha=0.6, label='Hand Onset' if onset == filtered_hand_onsets[0] else "")
    
    # Adding titles and labels
    plt.title('HiHat and Hand Voltage with Detected Onsets')
    plt.xlabel('Time [s]')
    plt.ylabel('Voltage [V]')
    plt.legend()
    plt.grid(True)
    
    # Show the plot
    plt.show()

# Testing the function with the full dataset
plot_onsets(data, hihat_onsets_result, hand_onsets_result)

# Optionally, we can zoom into a specific section
# plot_onsets(data, hihat_onsets_result, hand_onsets_result, zoom_start=40, zoom_end=41)



# %%
# Define a function to find onset pairs and ensure the output is (hihat, hand) regardless of the base signal
def find_hihat_hand_pairs(hihat_onsets, hand_onsets, max_diff=0.3):
    # Determine the base signal (the one with fewer onsets)
    if len(hihat_onsets) <= len(hand_onsets):
        base_onsets = hihat_onsets
        other_onsets = hand_onsets
        base_is_hihat = True  # Flag to indicate that the base signal is HiHat
    else:
        base_onsets = hand_onsets
        other_onsets = hihat_onsets
        base_is_hihat = False  # Flag to indicate that the base signal is Hand
    
    # Pair the base onsets with the closest onset from the other signal
    hihat_hand_pairs = []
    for base_onset in base_onsets:
        closest_other_onset = min(other_onsets, key=lambda x: abs(base_onset - x))
        if abs(base_onset - closest_other_onset) > max_diff:
            continue
        if base_is_hihat:
            hihat_hand_pairs.append((base_onset, closest_other_onset))  # (HiHat, Hand)
        else:
            hihat_hand_pairs.append((closest_other_onset, base_onset))  # (HiHat, Hand)
    
    return hihat_hand_pairs

hihat_hand_pairs_corrected = find_hihat_hand_pairs(hihat_onsets_result, hand_onsets_result)
pairs_df = pd.DataFrame(hihat_hand_pairs_corrected, columns=['HiHat_Onset', 'Hand_Onset'])
pairs_df.head()
# Testing the function

# Display the number of pairs and the first few pairs
len(hihat_hand_pairs_corrected), hihat_hand_pairs_corrected[:5]  # Show total pairs and the first 5 pairs (HiHat, Hand)


# %%
import TE
te_calculator = TE.TransferEntropyCalculator(jar_location='infodynamics-dist-1.6.1/infodynamics.jar', knns=4, dest_past_intervals=[1], source_past_intervals=[1], jittered_sampling=False)

time_points, te_dynamics = te_calculator.compute_TE_moving_window(pairs_df['Hand_Onset'].to_numpy(), pairs_df['HiHat_Onset'].to_numpy(), window_size=25, step_size=1)

# Print TE dynamics
for t, te in zip(time_points, te_dynamics):
    print(f"Time {t:.2f}: TE = {te:.4f}")


# plot the TE dynamics
plt.figure(figsize=(12, 6))
plt.plot(time_points, te_dynamics, label='Transfer Entropy', color='blue')
plt.title('Transfer Entropy Dynamics')
plt.xlabel('Time [s]')
plt.ylabel('Transfer Entropy')
plt.legend()
plt.grid(True)
plt.show()

# %%
# Make a dataframe from the pairs


# %%
# Plot the data for 30th second

# import matplotlib
# matplotlib.use('tkagg')
# import matplotlib.pyplot as plt
# plt.ion()
# %matplotlib tk
# plot_onsets(data, hihat_onsets_result, hand_onsets_result, zoom_start=25, zoom_end=35)
plot_onsets(data, hihat_onsets_result, hand_onsets_result)
# %%
# compute row differences using pairs_df.diff()
pairs_df_diff = pairs_df.diff()

# compute SD with moving window
window_size = 15
pairs_df_diff['HiHat_Onset_SD'] = pairs_df_diff['HiHat_Onset'].rolling(window=window_size).std()
pairs_df_diff['Hand_Onset_SD'] = pairs_df_diff['Hand_Onset'].rolling(window=window_size).std()
pairs_df_diff.head(25)

# %%
# plot the SD
plt.figure(figsize=(12, 6))
plt.plot(pairs_df_diff['HiHat_Onset_SD'], label='HiHat Onset SD', color='blue')
plt.plot(pairs_df_diff['Hand_Onset_SD'], label='Hand Onset SD', color='orange')
plt.title('HiHat and Hand Onset SD with Moving Window')
plt.xlabel('Pair Index')
plt.ylabel('Onset Time Difference [s]')
plt.legend()
plt.grid(True)
plt.show() 

# %%
pairs_df_ = pairs_df.copy()
pairs_df_['onset_diff'] = pairs_df['Hand_Onset'] - pairs_df['HiHat_Onset']
pairs_df_.head()

# %%
# plot the onset difference
plt.figure(figsize=(12, 6))
plt.plot(pairs_df_['onset_diff'], label='Onset Difference', color='blue')
plt.title('Onset Difference between HiHat and Hand')
plt.xlabel('Pair Index')
plt.ylabel('Onset Time Difference [s]')
plt.legend()
plt.grid(True)
plt.show()






# %%
'''
Reconstruct the time series with onset data. for onset time, set the value to 1, otherwise 0
'''
def reconstruct_time_series(data, hihat_onsets, hand_onsets):
    # Create a new DataFrame with the same time points as the original data
    reconstructed_data = pd.DataFrame({'Time[s]': data['Time[s]']})
    
    # Add columns for HiHat and Hand onsets
    reconstructed_data['HiHat_Onset'] = 0
    reconstructed_data['Hand_Onset'] = 0
    
    # Set the onset values to 1
    for onset in hihat_onsets:
        index = reconstructed_data[reconstructed_data['Time[s]'] == onset].index[0]
        reconstructed_data.at[index, 'HiHat_Onset'] = 1
    
    for onset in hand_onsets:
        index = reconstructed_data[reconstructed_data['Time[s]'] == onset].index[0]
        reconstructed_data.at[index, 'Hand_Onset'] = 1
    
    return reconstructed_data

# Testing the function
reconstructed_data = reconstruct_time_series(data, hihat_onsets_result, hand_onsets_result)
reconstructed_data.head()

# %%
# Plot the reconstructed time series
plt.figure(figsize=(12, 6))
plt.plot(reconstructed_data['Time[s]'], reconstructed_data['HiHat_Onset'], label='HiHat Onset', color='blue')
plt.plot(reconstructed_data['Time[s]'], reconstructed_data['Hand_Onset'], label='Hand Onset', color='orange')
plt.title('Reconstructed Time Series with Onsets')
plt.xlabel('Time [s]')
plt.ylabel('Onset')
plt.legend()
plt.grid(True)
plt.show()

# %%
'''
Convolving the onset data with a exponential window. THe window is defined as a function of time constant tau
'''
def convolve_with_exp_window(data, tau):
    # Define the time points and the window
    time_points = data['Time[s]']
    window = np.exp(-time_points / tau)
    
    # Convolve the HiHat and Hand onset data with the window
    hi_hat_onset_conv = np.convolve(data['HiHat_Onset'], window, mode='same')
    hand_onset_conv = np.convolve(data['Hand_Onset'], window, mode='same')
    
    return hi_hat_onset_conv, hand_onset_conv

# Testing the function
tau = 0.1
hi_hat_onset_conv, hand_onset_conv = convolve_with_exp_window(reconstructed_data, tau)

# Plot the convolved data
plt.figure(figsize=(12, 6))
plt.plot(reconstructed_data['Time[s]'], hi_hat_onset_conv, label='HiHat Onset Convolved', color='blue')
plt.plot(reconstructed_data['Time[s]'], hand_onset_conv, label='Hand Onset Convolved', color='orange')
plt.title('Convolved Time Series with Exponential Window')
plt.xlabel('Time [s]')
plt.ylabel('Onset')
plt.legend()
plt.grid(True)
plt.show()


# %%
ewma_binary_signal = reconstructed_data.ewm(span=100, adjust=False).mean()
ewma_binary_signal.head()

# %%
# Plot the convolved data
plt.figure(figsize=(12, 6))
plt.plot(reconstructed_data['Time[s]'], ewma_binary_signal['HiHat_Onset'], label='HiHat Onset Convolved', color='blue')
plt.plot(reconstructed_data['Time[s]'], ewma_binary_signal['Hand_Onset'], label='Hand Onset Convolved', color='orange')
plt.title('Convolved Time Series with Exponential Window')
plt.xlabel('Time [s]')
plt.ylabel('Onset')
plt.legend()
plt.grid(True)
plt.show()


#%% 
# save ewma_binary_signal to file
ewma_binary_signal.to_csv('data/' + file_path.split('/')[1].split('.')[0] + '_ewma.csv', index=False)

# $$
# save ewma_binary_signal without first column to file and without header 
ewma_binary_signal.iloc[:, 1:].to_csv('data/' + file_path.split('/')[1].split('.')[0] + '_ewma_binary.csv', index=False, header=False)


# %% 
from PyIF import te_compute as te
te.te_compute(ewma_binary_signal['HiHat_Onset'].to_numpy(), ewma_binary_signal['Hand_Onset'].to_numpy (), k=3, embedding=100, safetyCheck=False, GPU=True)
                              
                              

# %%
# compute average difference from pairs_df_diff
pairs_df_diff['HiHat_Onset'].mean(), pairs_df_diff['Hand_Onset'].mean()



# %%
diff = ewma_binary_signal['Time[s]'].diff()
# plot difference
plt.figure(figsize=(12, 6))
plt.plot(diff, label='Time Difference', color='blue')
plt.title('Time Difference')
plt.xlabel('Pair Index')
plt.ylabel('Time Difference [s]')
plt.legend()
plt.grid(True)
plt.show()

# %% 
from jpype import *

# %% 
%matplotlib tk

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


def time_normalize(timesteps):
    """
    Normalize the timestamps to 0 to 1
    """
    return (timesteps - timesteps[0]) / (timesteps[-1] - timesteps[0])

# Example usage:
# Just call the function and update the folder path inside it
parsed_df = parse_filenames_in_folder(filepath_raw)

# Display the parsed DataFrame
print(parsed_df)


# %%

# Define the file path or name
# file_path = 'raw/J_240718_LEDvib1.txt'
file_path = 'raw/A_240627_LEDvib4.txt'

# Load the data using pandas read_csv
data = pd.read_csv(file_path, 
                   delim_whitespace=True,    # This tells pandas that the delimiter is whitespace
                   header=0)                 # Use the first line as the header

# if the file has 5 columns, it is the drum data
if data.shape[1] == 5:
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
def detect_onsets(data, hihat_threshold=0.5, hand_threshold=0.1, ignore_duration_hihat=0.3, ignore_duration_hand=0.2):
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
len(hihat_onsets_result), len(hand_onsets_result), hihat_onsets_result[:5], hand_onsets_result[:5]  


# %%
# save the onset data to json file
import json
onset_dict = {'hihat_onsets': hihat_onsets_result, 'hand_onsets': hand_onsets_result}
with open('data/' + file_path.split('/')[1].split('.')[0] + '_onsets.json', 'w') as fp:
    json.dump(onset_dict, fp)


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
    # Always use hand_onsets as the base_onsets
    base_onsets = hand_onsets
    other_onsets = hihat_onsets
    
    # Pair the base onsets with the closest onset from the other signal
    hihat_hand_pairs = []
    skip_count = 0
    skip_onsets = []
    for base_onset in base_onsets:
        closest_other_onset = min(other_onsets, key=lambda x: abs(base_onset - x))
        if abs(base_onset - closest_other_onset) > max_diff:
            skip_count += 1
            skip_onsets.append(base_onset)
            continue
        hihat_hand_pairs.append((closest_other_onset, base_onset))  # (HiHat, Hand)
    
    return hihat_hand_pairs, skip_count, skip_onsets

hihat_hand_pairs, skip_count, skip_onsets = find_hihat_hand_pairs(hihat_onsets_result, hand_onsets_result)
pairs_df = pd.DataFrame(hihat_hand_pairs, columns=['HiHat_Onset', 'Hand_Onset'])
pairs_df.head()
# Testing the function

# print the number of pairs and skipped onsets
print('Number of pairs:', len(hihat_hand_pairs))
print('Number of skipped onsets:', skip_count)
print('Skipped onsets:', skip_onsets)



# %%
import TE
te_calculator = TE.TransferEntropyCalculator(jar_location='infodynamics-dist-1.6.1/infodynamics.jar', knns=3, dest_past_intervals=[1], source_past_intervals=[1], jittered_sampling=False)

time_points, te_dynamics = te_calculator.compute_TE_moving_window(pairs_df['Hand_Onset'].to_numpy(), pairs_df['HiHat_Onset'].to_numpy(), window_size=25, step_size=0.2)

# Print TE dynamics
for t, te in zip(time_points, te_dynamics):
    print(f"Time {t:.2f}: TE = {te:.4f}")


# plot the TE dynamics
plt.figure(figsize=(12, 6))
plt.plot(time_normalize(time_points), te_dynamics, label='Transfer Entropy', color='blue')
plt.title('Transfer Entropy Dynamics')
plt.xlabel('Normalised Time')
plt.ylabel('Transfer Entropy')
plt.legend()
plt.grid(True)
plt.show()

# save figure
plt.savefig('data/' + file_path.split('/')[1].split('.')[0] + '_TE.png')


# %%
# smooth the TE dynamics
window_size = 15
te_dynamics_smoothed = pd.Series(te_dynamics).rolling(window=window_size).mean()

# plot the smoothed TE dynamics
plt.figure(figsize=(12, 6))
plt.plot(time_normalize(time_points), te_dynamics_smoothed, label='Transfer Entropy', color='blue')
plt.title('Smoothed Transfer Entropy Dynamics')
plt.xlabel('Normalised Time')
plt.ylabel('Transfer Entropy')
plt.legend()
plt.grid(True)
plt.show()

# save figure
plt.savefig('data/' + file_path.split('/')[1].split('.')[0] + '_TE_smoothed.png')


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
# Normalize the time for the x-axis
normalized_time = time_normalize(pairs_df.index.to_numpy())

plt.plot(normalized_time, pairs_df_diff['HiHat_Onset_SD'], label='HiHat Onset SD', color='blue')
plt.plot(normalized_time, pairs_df_diff['Hand_Onset_SD'], label='Hand Onset SD', color='orange')
plt.title('HiHat and Hand Onset SD with Moving Window')
plt.xlabel('Normalised Time')
plt.ylabel('Onset Time Difference [s]')
plt.legend()
plt.grid(True)
plt.show() 

# save figure
plt.savefig('data/' + file_path.split('/')[1].split('.')[0] + '_SD.png')

# %%
pairs_df_ = pairs_df.copy()
pairs_df_['onset_diff'] = pairs_df['Hand_Onset'] - pairs_df['HiHat_Onset']
pairs_df_.head()

# %%
# plot the onset difference
normalized_time = time_normalize(pairs_df.index.to_numpy())
plt.figure(figsize=(12, 6))
plt.plot(normalized_time, pairs_df_['onset_diff'], label='Onset Difference', color='blue')
plt.axhline(y=0, color='red', linestyle='--', label='Zero Difference')
plt.title('Onset Difference between HiHat and Hand')
plt.xlabel('Normalised Time')
plt.ylabel('Onset Time Difference [s]')
plt.legend()
plt.grid(True)
plt.show()

# save figure
plt.savefig('data/' + file_path.split('/')[1].split('.')[0] + '_onset_diff.png')
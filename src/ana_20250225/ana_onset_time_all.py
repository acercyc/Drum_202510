"""
This script runs onset detection for multiple participants and trials, comparing reference onsets
with target onsets, plotting time differences, and saving results to disk. It automatically scans
available files, processes each, and stores summary data for further analysis.
"""
# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import participant_ids, raw_data_path, find_trials, import_soa_rating_data, import_emg_data

participant = participant_ids[0]
data_beh = import_soa_rating_data(participant)
trials = find_trials(participant)

trial = trials[0]
data_emg = import_emg_data(participant, trial)





# %%
raw_data_path = "../../data/raw_20241030/output_txt/"
save_data_path = "../../data/ana_20250114/"
participant_ids = ["datahayase1030", "datahosokawa1030", "datasawada1030", "datasugata1030"]
trial_ids = [str(i) for i in range(1, 11)]

def run_onset_detection_for_all():
    results = []
    for participant_id in participant_ids:
        for trial_id in trial_ids:
            file_name = f"{participant_id}_{trial_id}.txt"
            file_path = os.path.join(raw_data_path, file_name)
            if not os.path.exists(file_path):
                continue
            
            data = import_data(file_path)
            ref_onset = extract_onset_time(data, 'Hi-hat (cue)', threshold=0.1, minimal_interval=1000)
            tgt_onset = extract_onset_time(data, 'Hi-hat', threshold=0.1, minimal_interval=1000)
            _, time_diff = find_nearest_onset_time(tgt_onset, ref_onset)
            
            # ...existing code for plotting, etc...
            plt.figure(figsize=(10, 3))
            plt.plot(tgt_onset, time_diff, '-o', label="Time Differences")
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            plt.title(f"{participant_id}_{trial_id} Onset Time Differences")
            plt.xlabel("Target Onset Time (s)")
            plt.ylabel("Difference (s)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_data_path, f"{participant_id}_{trial_id}_onset_diff.png"))
            plt.close()
            
            results.append({
                "participant": participant_id,
                "trial": trial_id,
                "time_diff": time_diff
            })
    
    # Save results as CSV or pickle
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(save_data_path, "all_onset_detections.csv"), index=False)

if __name__ == "__main__":
    run_onset_detection_for_all()
# %%

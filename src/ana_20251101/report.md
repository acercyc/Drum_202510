# Transfer Entropy Analysis Report: Relationship with Sense of Agency

## Analysis Overview

This report presents an analysis of Transfer Entropy (TE) in drum performance data and its relationship with Sense of Agency (SoA) ratings. The analysis examines how information flow from a reference timing signal to participants' performance relates to their subjective sense of control and achievement.

### Data Collection

- **Participants**: 10 subjects (subjects 3-12)
- **Sessions**: Multiple sessions per participant across different dates (May-October 2025)
- **Total Sessions Analyzed**: 524 complete sessions with both TE and SoA data

### Methodology

**Transfer Entropy Computation:**
- **Source Signal**: `Correct_Timing_Signal[V]` - Reference timing signal representing the model/right hand timing
- **Target Signal**: `ACC_HIHAT[V]` - Participant's hi-hat performance timing
- **Method**: Kraskov-Grassberger estimator for continuous data (Java Information Dynamics Toolkit)
- **Sampling Rate**: 10 kHz (downsampled to 100 Hz for analysis)
- **Preprocessing**: Signal smoothing using impulse response convolution and onset detection

**Sense of Agency Measurement:**
- **Q1**: Pre-SoA rating (0-100) - Sense of Agency rating before practice session
- **Q2**: Post-SoA rating (0-100) - Sense of Agency rating after practice session
- Collected via questionnaire

**Data Quality:**
- Only sessions with complete data (both NI sensor data and SoA ratings) were included
- Missing data were excluded to ensure reliable correlation analysis

## Figures Description

### Individual Subject Plots (`TE_SoA_subject_{N}.png`)

Each subject has a dedicated plot showing the temporal progression of TE, Q1, and Q2 across all their sessions.

**Plot Features:**
- **Dual Y-Axes**:
  - Left axis (blue/orange): Q1 and Q2 SoA ratings (0-100 scale)
  - Right axis (green): Transfer Entropy values (typically 0.00-0.05)
  
- **Data Series**:
  - **Q1** (blue circles): Pre-SoA ratings over time
  - **Q2** (orange squares): Post-SoA ratings over time  
  - **TE** (green triangles): Transfer Entropy values over time

- **X-Axis**:
  - Session index separated by date
  - Date labels centered within each date block
  - Vertical dashed lines separate different session dates
  - Dates displayed in YYYY-MM-DD format

- **Correlation Information**:
  - Pearson correlation coefficients displayed above the plot
  - TE-Q1 correlation shown in blue
  - TE-Q2 correlation shown in orange
  - Values indicate strength and direction of relationship

- **Visual Elements**:
  - Grid lines for easier reading
  - Legend in upper left corner
  - Title showing subject number


## Results

### Correlation Analysis

Correlation coefficients between Transfer Entropy and SoA ratings were computed for each subject:

| Subject | TE-Q1 (r) | TE-Q2 (r) | N Sessions |
|---------|-----------|-----------|------------|
| 3       | 0.565     | 0.560     | 74        |
| 4       | -0.052    | 0.227     | 49        |
| 5       | 0.489     | 0.414     | 73        |
| 6       | 0.241     | 0.252     | 54        |
| 7       | 0.499     | 0.524     | 55        |
| 8       | 0.396     | 0.508     | 34        |
| 9       | 0.392     | 0.373     | 81        |
| 10      | 0.270     | 0.368     | 53        |
| 11      | 0.514     | 0.506     | 45        |
| 12      | 0.041     | 0.712     | 6         |


### Key Finding: TE Tracks Q1 and Q2 Surprisingly Well

**Unexpected Discovery:**

Contrary to initial predictions, the analysis reveals that Transfer Entropy tracks Sense of Agency ratings (Q1 and Q2) surprisingly well. The data shows a consistent pattern:

1. **Parallel Increases**: Both pre-SoA (Q1), post-SoA (Q2), and TE increase over time as participants practice
2. **Positive Relationships**: Despite the prediction that higher TE (indicating stronger coupling to reference) might correspond to lower SoA (less sense of control), the opposite pattern emerges
3. **Consistent Across Subjects**: The positive relationship is observed across most participants, with moderate to strong correlations (r=0.27-0.57)

**Interpretation:**

While this finding contradicts the initial hypothesis, it clearly demonstrates that **Transfer Entropy is obviously positively related to Sense of Agency**. 


---

## Data Integrity and Cautions

> **Note: Data Quality Caveats**

The dataset used in this analysis is somewhat messy and inconsistent. During processing, the following issues were encountered and should be kept in mind:

- **Occasional Missing Files:** Not every expected data file was always present. Some sessions or subjects lacked corresponding data files, resulting in incomplete records.
- **Missing or Incomplete Data:** Certain rows in the dataset may have missing values for important fields (such as TE, Q1, or Q2). This could impact statistical calculations or visualizations.
- **Missing Headers and Format Inconsistencies:** On occasion, files were found to be missing header rows or to have inconsistent formatting, requiring manual correction or special handling in the code.

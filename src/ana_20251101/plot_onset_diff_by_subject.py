"""
Plot Mean Onset Difference (mean_diff), Q1, and Q2 grouped by subject.

For each subject, creates a plot showing:
- Mean onset difference (action error) values over time (dates)
- Q1 (pre-SoA) and Q2 (post-SoA) ratings over time
- Vertical lines separating different dates
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

# Set font to handle CJK characters (Japanese months)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def load_clean_data(csv_path: str = "onset_diff_results_all.csv") -> pd.DataFrame:
    """
    Load the cleaned onset difference results CSV.
    
    Args:
        csv_path: Path to cleaned CSV file
        
    Returns:
        DataFrame with cleaned data
    """
    csv_file = Path(csv_path)
    
    if not csv_file.exists():
        # Try relative to script directory
        script_dir = Path(__file__).parent
        csv_file = script_dir / csv_path
        if not csv_file.exists():
            raise FileNotFoundError(f"Clean data file not found: {csv_path}")
    
    print(f"Loading clean data from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Convert date to datetime for proper sorting
    if 'date' in df.columns:
        df['date_str'] = df['date'].astype(str)
        # Convert date string (YYYYMMDD) to datetime
        df['date_dt'] = pd.to_datetime(df['date_str'], format='%Y%m%d', errors='coerce')
    
    # Convert datetime column to datetime if it exists
    if 'datetime' in df.columns:
        df['datetime_dt'] = pd.to_datetime(df['datetime'], errors='coerce')
    
    print(f"Loaded {len(df)} rows")
    print(f"Subjects: {sorted(df['subject'].unique())}")
    
    return df


def plot_subject_data(df: pd.DataFrame, subject: str, save_dir: Path = None, show_plot: bool = True):
    """
    Plot mean_diff, Q1, and Q2 for a single subject.
    
    Args:
        df: DataFrame with all data
        subject: Subject ID (as string)
        save_dir: Directory to save plot (if None, doesn't save)
        show_plot: Whether to display the plot
    """
    # Filter data for this subject
    subject_data = df[df['subject'].astype(str) == str(subject)].copy()
    
    if len(subject_data) == 0:
        print(f"No data found for subject {subject}")
        return
    
    # Sort by date
    subject_data = subject_data.sort_values('date_dt')
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()
    
    # Get unique dates for vertical line separation
    unique_dates = subject_data['date_dt'].unique()
    unique_dates = sorted([d for d in unique_dates if pd.notna(d)])
    
    # Create x-axis positions (use index within each date)
    x_positions = []
    date_boundaries = []  # Store (start, end, middle) for each date
    current_pos = 0
    
    for date in unique_dates:
        date_data = subject_data[subject_data['date_dt'] == date]
        date_start = current_pos
        
        for idx in date_data.index:
            x_positions.append(current_pos)
            current_pos += 1
        
        date_end = current_pos - 1
        date_middle = (date_start + date_end) / 2
        
        # Store boundaries: (start, end, middle, date)
        date_boundaries.append({
            'start': date_start,
            'end': date_end,
            'middle': date_middle,
            'date': date
        })
    
    x_positions = np.array(x_positions)
    
    # Create a mapping from subject_data index to x_position index
    index_to_xpos = {idx: x_pos for idx, x_pos in zip(subject_data.index, x_positions)}
    
    # Plot Q1 and Q2 on left y-axis
    if 'Q1' in subject_data.columns and 'Q2' in subject_data.columns:
        q1_valid = subject_data['Q1'].notna()
        q2_valid = subject_data['Q2'].notna()
        
        if q1_valid.any():
            q1_indices = subject_data.index[q1_valid]
            q1_xpos = [index_to_xpos[idx] for idx in q1_indices]
            ax1.plot(q1_xpos, subject_data.loc[q1_indices, 'Q1'].values,
                    'o-', label='Q1', color='tab:blue', markersize=6, linewidth=2, alpha=0.7)
        
        if q2_valid.any():
            q2_indices = subject_data.index[q2_valid]
            q2_xpos = [index_to_xpos[idx] for idx in q2_indices]
            ax1.plot(q2_xpos, subject_data.loc[q2_indices, 'Q2'].values,
                    's-', label='Q2', color='tab:orange', markersize=6, linewidth=2, alpha=0.7)
    
    # Plot mean_diff on right y-axis
    if 'mean_diff' in subject_data.columns:
        mean_diff_valid = subject_data['mean_diff'].notna()
        if mean_diff_valid.any():
            mean_diff_indices = subject_data.index[mean_diff_valid]
            mean_diff_xpos = [index_to_xpos[idx] for idx in mean_diff_indices]
            ax2.plot(mean_diff_xpos, subject_data.loc[mean_diff_indices, 'mean_diff'].values,
                    '^-', label='Mean Onset Diff', color='tab:green', markersize=6, linewidth=2, alpha=0.7)
    
    # Add vertical lines to separate dates (between date blocks)
    for i in range(len(date_boundaries) - 1):
        # Vertical line at the boundary between dates
        boundary_pos = date_boundaries[i]['end'] + 0.5
        ax1.axvline(x=boundary_pos, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Set labels and title
    ax1.set_xlabel('Session Index (separated by date)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Q1 (pre-SoA) & Q2 (post-SoA)', fontsize=12, fontweight='bold', color='tab:blue')
    ax2.set_ylabel('Mean Onset Difference (s)', fontsize=12, fontweight='bold', color='tab:green')
    
    # Color the y-axis labels
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    
    # Format x-axis with date labels - center them in their date blocks
    x_ticks = []
    x_labels = []
    
    for boundary_info in date_boundaries:
        # Use the middle position of each date's data block
        x_ticks.append(boundary_info['middle'])
        # Format date as YYYY-MM-DD
        date_str = boundary_info['date'].strftime('%Y-%m-%d')
        x_labels.append(date_str)
    
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right')
    
    # Set x-axis limits with some padding
    ax1.set_xlim(-0.5, len(x_positions) - 0.5)
    
    # Calculate correlations
    correlations = {}
    
    if 'mean_diff' in subject_data.columns and 'Q1' in subject_data.columns:
        # mean_diff vs Q1 correlation
        valid_data = subject_data[['mean_diff', 'Q1']].dropna()
        if len(valid_data) > 1:
            corr_q1 = valid_data['mean_diff'].corr(valid_data['Q1'])
            correlations['Q1'] = corr_q1
    
    if 'mean_diff' in subject_data.columns and 'Q2' in subject_data.columns:
        # mean_diff vs Q2 correlation
        valid_data = subject_data[['mean_diff', 'Q2']].dropna()
        if len(valid_data) > 1:
            corr_q2 = valid_data['mean_diff'].corr(valid_data['Q2'])
            correlations['Q2'] = corr_q2
    
    # Title
    month = subject_data['month'].iloc[0] if 'month' in subject_data.columns else 'Unknown'
    title = f'Subject {subject} - Mean Onset Difference (Action Error) and SoA Ratings'
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    # Add correlation text above the plot frame
    if correlations:
        # Place above the plot, outside the frame, right-aligned
        # Use figure coordinates instead of axes coordinates
        fig = ax1.figure
        y_position = 1.02  # Above the axes (1.0 is top of axes)
        
        if 'Q1' in correlations:
            # Q1 (pre-SoA) correlation with blue color for label
            fig.text(0.98, y_position, 
                    f"MeanDiff-Q1 (pre-SoA): r={correlations['Q1']:.3f}", 
                    fontsize=11,
                    verticalalignment='bottom',
                    horizontalalignment='right',
                    color='tab:blue',  # Match Q1 line color
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.5),
                    family='monospace',
                    transform=fig.transFigure)
            y_position += 0.05
        
        if 'Q2' in correlations:
            # Q2 (post-SoA) correlation with orange color for label
            fig.text(0.98, y_position, 
                    f"MeanDiff-Q2 (post-SoA): r={correlations['Q2']:.3f}", 
                    fontsize=11,
                    verticalalignment='bottom',
                    horizontalalignment='right',
                    color='tab:orange',  # Match Q2 line color
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.5),
                    family='monospace',
                    transform=fig.transFigure)
    
    # Grid
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save if requested
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        output_file = save_dir / f"onset_diff_SoA_subject_{subject}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Saved to: {output_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_all_subjects(df: pd.DataFrame, save_dir: str = None, show_plots: bool = False):
    """
    Plot data for all subjects.
    
    Args:
        df: DataFrame with all data
        save_dir: Directory to save plots (if None, uses 'plots_onset_diff' subdirectory)
        show_plots: Whether to display plots
    """
    subjects = sorted(df['subject'].astype(str).unique())
    
    if save_dir:
        save_path = Path(save_dir)
    else:
        save_path = Path(__file__).parent / "plots_onset_diff"
    
    print(f"\nPlotting {len(subjects)} subjects...")
    print(f"Plots will be saved to: {save_path}")
    
    for subject in subjects:
        print(f"\nPlotting subject {subject}...")
        plot_subject_data(df, subject, save_dir=save_path, show_plot=show_plots)


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Plot Mean Onset Difference, Q1, and Q2 grouped by subject"
    )
    parser.add_argument(
        "--input", type=str, default="onset_diff_results_all.csv",
        help="Input CSV file (default: onset_diff_results_all.csv)"
    )
    parser.add_argument(
        "--subject", type=str,
        help="Plot specific subject only (if not specified, plots all subjects)"
    )
    parser.add_argument(
        "--save-dir", type=str, default="plots_onset_diff",
        help="Directory to save plots (default: plots_onset_diff)"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display plots (default: False, only save)"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save plots, only display"
    )
    
    args = parser.parse_args()
    
    # Load data
    df = load_clean_data(args.input)
    
    # Plot
    if args.subject:
        # Plot single subject
        save_dir = None if args.no_save else Path(args.save_dir)
        plot_subject_data(df, args.subject, save_dir=save_dir, show_plot=args.show)
    else:
        # Plot all subjects
        save_dir = None if args.no_save else args.save_dir
        plot_all_subjects(df, save_dir=save_dir, show_plots=args.show)


if __name__ == "__main__":
    main()


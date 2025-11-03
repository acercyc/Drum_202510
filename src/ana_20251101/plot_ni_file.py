"""
Plot data from an NI file.

This script loads an NI file and visualizes the sensor signals,
with special focus on Correct_Timing_Signal[V] and ACC_HIHAT[V].
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys


def plot_ni_file(ni_file_path, save_plot=False, output_path=None):
    """
    Plot all signals from an NI file.
    
    Args:
        ni_file_path: Path to NI file (tab-separated CSV)
        save_plot: Whether to save the plot to file
        output_path: Path to save plot (if None, uses ni_file_path with .png extension)
    """
    # Load NI data
    print(f"Loading NI file: {ni_file_path}")
    data = pd.read_csv(ni_file_path, sep="\t", header=0)
    
    print(f"Shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"Time range: {data['Time[s]'].min():.2f} - {data['Time[s]'].max():.2f} seconds")
    print(f"Duration: {data['Time[s]'].max() - data['Time[s]'].min():.2f} seconds")
    
    time = data['Time[s]'].values
    
    # Create figure with subplots
    n_cols = len(data.columns) - 1  # Exclude Time column
    n_rows = (n_cols + 2) // 3  # 3 columns per row
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
    
    # Plot each signal
    for idx, col in enumerate(data.columns):
        if col == 'Time[s]':
            continue
        
        ax = axes[idx - 1] if idx > 0 else axes[0]
        
        signal = data[col].values
        
        # Highlight key signals
        if col == 'Correct_Timing_Signal[V]':
            ax.plot(time, signal, label=col, color='green', linewidth=1.5, alpha=0.8)
            ax.set_title(f'{col} (Reference Signal)', fontweight='bold', color='green')
        elif col == 'ACC_HIHAT[V]':
            ax.plot(time, signal, label=col, color='blue', linewidth=1.5, alpha=0.8)
            ax.set_title(f'{col} (Target Signal)', fontweight='bold', color='blue')
        else:
            ax.plot(time, signal, label=col, linewidth=1, alpha=0.7)
            ax.set_title(col)
        
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Voltage [V]')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for idx in range(len(data.columns) - 1, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save or show
    if save_plot:
        if output_path is None:
            ni_path = Path(ni_file_path)
            output_path = ni_path.parent / f"{ni_path.stem}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()
    
    return fig


def plot_key_signals(ni_file_path, save_plot=False, output_path=None):
    """
    Plot only the key signals for TE analysis: Correct_Timing_Signal and ACC_HIHAT.
    
    Args:
        ni_file_path: Path to NI file
        save_plot: Whether to save the plot
        output_path: Path to save plot
    """
    # Load NI data
    print(f"Loading NI file: {ni_file_path}")
    data = pd.read_csv(ni_file_path, sep="\t", header=0)
    
    time = data['Time[s]'].values
    
    # Extract key signals
    source = data['Correct_Timing_Signal[V]'].values
    target = data['ACC_HIHAT[V]'].values
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot source signal
    ax1.plot(time, source, color='green', linewidth=1.5, alpha=0.8)
    ax1.set_title('Correct_Timing_Signal[V] (Reference Signal - Source)', 
                  fontsize=14, fontweight='bold', color='green')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Voltage [V]')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    
    # Plot target signal
    ax2.plot(time, target, color='blue', linewidth=1.5, alpha=0.8)
    ax2.set_title('ACC_HIHAT[V] (Hi-hat Performance - Target)', 
                  fontsize=14, fontweight='bold', color='blue')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Voltage [V]')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    
    # Plot both signals together for comparison
    ax3.plot(time, source, label='Correct_Timing_Signal[V] (Source)', 
             color='green', linewidth=1.5, alpha=0.7)
    ax3.plot(time, target, label='ACC_HIHAT[V] (Target)', 
             color='blue', linewidth=1.5, alpha=0.7)
    ax3.set_title('Both Signals Overlaid (for Transfer Entropy Analysis)', 
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Voltage [V]')
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    # Save or show
    if save_plot:
        if output_path is None:
            ni_path = Path(ni_file_path)
            output_path = ni_path.parent / f"{ni_path.stem}_key_signals.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()
    
    return fig


def plot_zoom(ni_file_path, start_time=None, end_time=None, duration=10, save_plot=False):
    """
    Plot a zoomed-in view of the signals.
    
    Args:
        ni_file_path: Path to NI file
        start_time: Start time in seconds (if None, auto-selects interesting region)
        end_time: End time in seconds
        duration: Duration to plot if start_time is None
        save_plot: Whether to save the plot
    """
    # Load NI data
    data = pd.read_csv(ni_file_path, sep="\t", header=0)
    time = data['Time[s]'].values
    
    # Determine time range
    if start_time is None:
        # Find a region with activity (high variance)
        source = data['Correct_Timing_Signal[V]'].values
        window_size = int(duration * 10000)  # 10kHz sampling rate
        if len(source) > window_size:
            variances = []
            for i in range(0, len(source) - window_size, window_size // 2):
                var = np.var(source[i:i+window_size])
                variances.append((i, var))
            start_idx = max(variances, key=lambda x: x[1])[0]
            start_time = time[start_idx]
        else:
            start_time = time[0]
    
    if end_time is None:
        end_time = start_time + duration
    
    # Filter data
    mask = (time >= start_time) & (time <= end_time)
    time_zoom = time[mask]
    source_zoom = data['Correct_Timing_Signal[V]'].values[mask]
    target_zoom = data['ACC_HIHAT[V]'].values[mask]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot source
    ax1.plot(time_zoom, source_zoom, color='green', linewidth=2, alpha=0.8)
    ax1.set_title(f'Correct_Timing_Signal[V] (Zoomed: {start_time:.1f}s - {end_time:.1f}s)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Voltage [V]')
    ax1.grid(True, alpha=0.3)
    
    # Plot target
    ax2.plot(time_zoom, target_zoom, color='blue', linewidth=2, alpha=0.8)
    ax2.set_title(f'ACC_HIHAT[V] (Zoomed: {start_time:.1f}s - {end_time:.1f}s)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Voltage [V]')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        ni_path = Path(ni_file_path)
        output_path = ni_path.parent / f"{ni_path.stem}_zoomed.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nZoomed plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig


def main():
    """Main function with command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot data from an NI file")
    parser.add_argument("ni_file", type=str, help="Path to NI file")
    parser.add_argument("--all", action="store_true", help="Plot all signals")
    parser.add_argument("--key", action="store_true", default=True, help="Plot only key signals (default)")
    parser.add_argument("--zoom", action="store_true", help="Plot zoomed view")
    parser.add_argument("--start", type=float, help="Start time for zoom (seconds)")
    parser.add_argument("--end", type=float, help="End time for zoom (seconds)")
    parser.add_argument("--duration", type=float, default=10, help="Duration for auto-zoom (seconds)")
    parser.add_argument("--save", action="store_true", help="Save plot instead of showing")
    
    args = parser.parse_args()
    
    ni_file_path = Path(args.ni_file)
    if not ni_file_path.exists():
        print(f"Error: File not found: {ni_file_path}")
        return
    
    if args.all:
        plot_ni_file(ni_file_path, save_plot=args.save)
    elif args.zoom:
        plot_zoom(ni_file_path, start_time=args.start, end_time=args.end, 
                 duration=args.duration, save_plot=args.save)
    else:
        # Default: plot key signals
        plot_key_signals(ni_file_path, save_plot=args.save)


if __name__ == "__main__":
    # If run directly without arguments, use a sample file
    if len(sys.argv) == 1:
        # Try to find a sample NI file
        from locate_data_files import find_all_data_groups, default_base_dir
        
        base_dir = default_base_dir()
        groups = find_all_data_groups(base_dir)
        
        # Find first group with NI files
        sample_file = None
        for group in groups:
            if len(group.ni_files) > 0:
                sample_file = group.ni_files[0]
                break
        
        if sample_file:
            print(f"Using sample file: {sample_file}")
            print("Plotting key signals...\n")
            plot_key_signals(sample_file, save_plot=False)
        else:
            print("No NI files found. Please specify a file path.")
            print("\nUsage:")
            print("  python plot_ni_file.py <ni_file_path>")
            print("  python plot_ni_file.py <ni_file_path> --all      # Plot all signals")
            print("  python plot_ni_file.py <ni_file_path> --key     # Plot key signals (default)")
            print("  python plot_ni_file.py <ni_file_path> --zoom    # Plot zoomed view")
            print("  python plot_ni_file.py <ni_file_path> --save    # Save plot instead of showing")
    else:
        main()


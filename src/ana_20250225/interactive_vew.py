import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, RangeSlider


DEFAULT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "raw_20250911",
    "NI",
    "20250604",
    "1_20250604_135334_ni.txt",
)


def estimate_sampling_rate(time_array: np.ndarray) -> float:
    """
    Estimate sampling rate from a monotonically increasing time vector in seconds.

    Uses the median of first differences for robustness to occasional irregularities.
    """
    if time_array.size < 2:
        return float("nan")
    dt = np.diff(time_array)
    # guard against zeros/negatives if any header artifacts
    dt = dt[dt > 0]
    if dt.size == 0:
        return float("nan")
    fs = 1.0 / np.median(dt)
    return float(fs)


def load_data(file_path: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load NI-style tab-delimited text with first column as Time[s].

    Returns (time_seconds, data_frame).
    """
    df = pd.read_csv(file_path, sep="\t")
    if df.columns[0].lower().startswith("time"):
        t = df.iloc[:, 0].to_numpy(dtype=float)
        data = df.iloc[:, 1:]
    else:
        # fallback: assume first column is time
        t = df.iloc[:, 0].to_numpy(dtype=float)
        data = df.iloc[:, 1:]
    return t, data


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    window = int(window)
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x, kernel, mode="same")


def decimate_stride(x: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return x
    return x[::factor]


def build_viewer(file_path: str, show: bool = True) -> None:
    t, data = load_data(file_path)
    channels = list(data.columns)
    fs = estimate_sampling_rate(t)

    # Initial decimation and smoothing parameters
    decim_default = 10 if fs and fs > 5000 else 1
    smooth_default = 1

    # Prepare decimated versions for plotting
    def prepare_series(y: np.ndarray, decim: int, smooth: int) -> Tuple[np.ndarray, np.ndarray]:
        y_proc = moving_average(y, smooth)
        t_d = decimate_stride(t, decim)
        y_d = decimate_stride(y_proc, decim)
        return t_d, y_d

    # Figure layout
    try:
        plt.style.use("seaborn-v0_8")
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(left=0.12, right=0.88, bottom=0.2, top=0.92)

    # Plot all channels initially visible
    lines = []
    for ch in channels:
        td, yd = prepare_series(data[ch].to_numpy(dtype=float), decim_default, smooth_default)
        line, = ax.plot(td, yd, label=ch, linewidth=0.8)
        lines.append(line)

    ax.set_title(f"Interactive Viewer: {os.path.basename(file_path)}  |  fs≈{fs:.1f} Hz")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (V)")
    ax.grid(True, alpha=0.3)

    # Legend outside
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    # Range slider for time
    ax_slider = plt.axes([0.12, 0.08, 0.65, 0.05])
    t_min, t_max = float(t.min()), float(t.max())
    slider = RangeSlider(ax_slider, "Time", t_min, t_max, valinit=(t_min, t_min + min(t_max - t_min, 5.0)))

    # CheckButtons for channels
    btn_ax = plt.axes([0.88, 0.2, 0.1, 0.6])
    visibility = [True] * len(channels)
    checks = CheckButtons(btn_ax, channels, visibility)

    # Secondary controls: simple text boxes via key bindings
    state = {
        "decim": decim_default,
        "smooth": smooth_default,
    }

    info_text = (
        f"Keys: [+]/[-] decim={state['decim']}  |  [s]/[S] smooth={state['smooth']}  |  [r] reset view"
    )
    info = fig.text(0.12, 0.02, info_text, fontsize=9)

    # Update functions
    def update_plot():
        # recompute series for all visible channels with current decim/smooth
        for ch, line in zip(channels, lines):
            y = data[ch].to_numpy(dtype=float)
            td, yd = prepare_series(y, state["decim"], state["smooth"]) 
            line.set_data(td, yd)
        ax.relim()
        ax.autoscale_view()
        # keep x-limits consistent with slider selection
        xmin, xmax = slider.val
        ax.set_xlim(xmin, xmax)
        info.set_text(
            f"Keys: [+]/[-] decim={state['decim']}  |  [s]/[S] smooth={state['smooth']}  |  [r] reset view"
        )
        fig.canvas.draw_idle()

    def on_slider_change(val):
        xmin, xmax = slider.val
        ax.set_xlim(xmin, xmax)
        fig.canvas.draw_idle()

    def on_check(label):
        idx = channels.index(label)
        line = lines[idx]
        line.set_visible(not line.get_visible())
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "+":
            state["decim"] = max(1, state["decim"] // 2)
            update_plot()
        elif event.key == "-":
            state["decim"] = min(1000, state["decim"] * 2)
            update_plot()
        elif event.key == "s":
            state["smooth"] = max(1, state["smooth"] - 1)
            update_plot()
        elif event.key == "S":
            state["smooth"] = min(1001, state["smooth"] + 1)
            update_plot()
        elif event.key == "r":
            slider.set_val((t_min, t_max))
            ax.set_xlim(t_min, t_max)
            fig.canvas.draw_idle()

    slider.on_changed(on_slider_change)
    checks.on_clicked(on_check)
    fig.canvas.mpl_connect("key_press_event", on_key)

    if show:
        plt.show()


def main(argv: List[str] = None) -> None:
    parser = argparse.ArgumentParser(description="Interactive viewer for NI time-series text file")
    parser.add_argument("file", nargs="?", default=DEFAULT_FILE, help="Path to NI .txt file (tab-delimited)")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.file):
        print(f"File not found: {args.file}")
        sys.exit(1)

    build_viewer(args.file)


if __name__ == "__main__":
    main()

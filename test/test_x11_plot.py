#!/usr/bin/env python3
"""
Test script to verify X11 display functionality with matplotlib.
This script generates sample data and creates a plot to test X11 forwarding.
"""

import os
import numpy as np

# Unset headless-related environment variables that might prevent GUI backends
os.environ.pop('MPLBACKEND', None)

# Prevent Qt from detecting headless mode
if 'DISPLAY' in os.environ:
    # Set Qt platform to xcb (X11) if DISPLAY is available
    os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')

# Check if DISPLAY is set before importing matplotlib
display = os.environ.get('DISPLAY')
if display:
    print(f"DISPLAY environment variable: {display}")
else:
    print("WARNING: DISPLAY environment variable is not set!")
    print("X11 forwarding may not be working.")
    print("Try: export DISPLAY=:0.0 (or your X11 display)")

import matplotlib
# Try to set an X11-compatible backend BEFORE importing pyplot
# Try backends in order of preference for X11 forwarding
backends_to_try = ['Qt5Agg', 'Qt4Agg', 'GTK3Agg', 'GTKAgg', 'TkAgg']
backend_set = False

for backend in backends_to_try:
    try:
        matplotlib.use(backend, force=True)
        print(f"Attempting to use matplotlib backend: {backend}")
        backend_set = True
        break
    except (ImportError, ValueError) as e:
        print(f"  Backend {backend} not available: {e}")
        continue

# Import pyplot - handle errors that occur during import
# The error happens when pyplot tries to initialize the backend
plt = None
backend_works = False

# Try to import with the selected backend (GUI or default)
try:
    import matplotlib.pyplot as plt
    # Test if backend actually works by trying to create a figure
    _test_fig = plt.figure()
    plt.close(_test_fig)
    print(f"Successfully imported pyplot with backend: {matplotlib.get_backend()}")
    backend_works = True
except (ImportError, RuntimeError, Exception) as e:
    error_msg = str(e)
    if 'headless' in error_msg.lower() or 'backend' in error_msg.lower() or 'qt' in error_msg.lower() or 'tk' in error_msg.lower():
        print(f"Backend failed during import: {error_msg}")
        backend_works = False
    else:
        # Unknown error, but might still work
        print(f"Warning during backend test: {error_msg}")
        backend_works = True  # Assume it works if it's not a backend error

if not backend_works:
    # Fall back to Agg backend
    print("Falling back to non-interactive backend (Agg)")
    matplotlib.use('Agg', force=True)
    # Clear pyplot from cache if it was partially loaded
    import sys
    if 'matplotlib.pyplot' in sys.modules:
        del sys.modules['matplotlib.pyplot']
    import matplotlib.pyplot as plt
    print(f"Using backend: {matplotlib.get_backend()}")

# Ensure plt is always defined
if plt is None:
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt

def test_x11_plot():
    """Generate sample data and create a test plot."""
    global plt  # Declare plt as global at the start
    
    # Print final backend info
    current_backend = matplotlib.get_backend()
    print(f"\nFinal matplotlib backend: {current_backend}")
    
    # Check if we have a GUI backend
    gui_backends = ['Qt5Agg', 'Qt4Agg', 'GTK3Agg', 'GTKAgg', 'TkAgg', 'TkCairo']
    is_gui_backend = any(backend in current_backend for backend in gui_backends)
    
    # Generate sample data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.sin(x) * np.cos(x)
    
    # Create figure with multiple subplots
    # Wrap in try-except to catch backend errors during figure creation
    try:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    except (ImportError, RuntimeError) as e:
        error_msg = str(e)
        if 'headless' in error_msg.lower() or 'backend' in error_msg.lower():
            print(f"\nError creating plot with GUI backend: {error_msg}")
            print("Switching to non-interactive backend (Agg)")
            matplotlib.use('Agg', force=True)
            # Re-import pyplot after changing backend
            import sys
            if 'matplotlib.pyplot' in sys.modules:
                del sys.modules['matplotlib.pyplot']
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
            is_gui_backend = False
        else:
            raise
    fig.suptitle('X11 Display Test - Sample Data Plots', fontsize=14, fontweight='bold')
    
    # First subplot: sine and cosine
    axes[0].plot(x, y1, label='sin(x)', linewidth=2, color='blue')
    axes[0].plot(x, y2, label='cos(x)', linewidth=2, color='red')
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('y', fontsize=12)
    axes[0].set_title('Sine and Cosine Functions')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Second subplot: product
    axes[1].plot(x, y3, label='sin(x) * cos(x)', linewidth=2, color='green')
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('y', fontsize=12)
    axes[1].set_title('Product of Sine and Cosine')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Display the plot
    if is_gui_backend:
        print("\nDisplaying plot window...")
        print("If the plot window appears, X11 forwarding is working correctly!")
        print("Close the plot window to exit.\n")
        plt.show()
        print("Plot window closed. X11 test completed successfully!")
    else:
        # Save to file instead
        output_file = 'test_x11_plot_output.png'
        print(f"\nGUI backend not available. Saving plot to: {output_file}")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved successfully to {output_file}")
        print("\nNote: To display plots interactively, ensure:")
        print("  1. X11 forwarding is enabled (ssh -X)")
        print("  2. DISPLAY environment variable is set")
        print("  3. Required GUI libraries are installed (tk, qt5, gtk3, etc.)")

if __name__ == '__main__':
    try:
        test_x11_plot()
    except Exception as e:
        print(f"Error: {e}")
        print("\nPossible issues:")
        print("1. X11 forwarding not enabled in SSH")
        print("2. DISPLAY environment variable not set")
        print("3. X11 server not running")
        raise


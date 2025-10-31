import os
import numpy as np
from abc import ABC, abstractmethod
from jpype import *

# find the current script directory
from pathlib import Path

# locate jitd jar file
path_src = Path(__file__).parent.resolve()
jitd_jar_location_default = os.path.join(path_src, "infodynamics-dist-1.6.1/infodynamics.jar")
if not os.path.isfile(jitd_jar_location_default):
    exit(f"infodynamics.jar not found (expected at {os.path.abspath(jitd_jar_location_default)})")
    

class TransferEntropyCalculator(ABC):
    def __init__(self, jar_location=None):
        if jar_location is None:
            jar_location = jitd_jar_location_default
        if not isJVMStarted():
            startJVM(getDefaultJVMPath(), "-ea", f"-Djava.class.path={jar_location}")

        self.teCalcClass = self.initialise_teCalcClass()
        
    @abstractmethod
    def initialise_teCalcClass(self):
        raise NotImplementedError("This method should be implemented by subclasses")
    
    @abstractmethod
    def compute_TE(self, source_array, destination_array):
        pass
    
    def compute_TE_moving_window(self, source_array, destination_array, window_size, step_size, **kwargs):
        te_values = []
        time_points = []
        min_time = min(np.min(source_array), np.min(destination_array))
        max_time = max(np.max(source_array), np.max(destination_array))
        current_start = min_time
        current_end = current_start + window_size
        while current_end <= max_time:
            source_window = source_array[(source_array >= current_start) & (source_array < current_end)]
            dest_window = destination_array[(destination_array >= current_start) & (destination_array < current_end)]
            if len(source_window) > 0 and len(dest_window) > 0:
                te_value = self.compute_TE(source_window, dest_window, **kwargs)
                te_values.append(te_value)
            else:
                te_values.append(np.nan)
            time_points.append((current_start + current_end) / 2)
            current_start += step_size
            current_end += step_size
        return time_points, te_values
    
    def shutdown(self):
        if isJVMStarted():
            shutdownJVM()
    

class TransferEntropyCalculator_continuous:
    

# class TransferEntropyCalculator_spiking:
#     def __init__(self, jar_location, knns=4, dest_past_intervals=[1, 2], source_past_intervals=[1, 2], jittered_sampling=False):
#         """
#         Initialize the TransferEntropyCalculator class and set the default parameters for TE calculation.
        
#         Args:
#             jar_location (str): Path to the JIDT jar file.
#             knns (int): Number of nearest neighbors for TE computation.
#             dest_past_intervals (list): List of past intervals for destination.
#             source_past_intervals (list): List of past intervals for source.
#             jittered_sampling (bool): Whether to use jittered sampling.
#         """
#         self.knns = knns
#         self.dest_past_intervals = dest_past_intervals
#         self.source_past_intervals = source_past_intervals
#         self.jittered_sampling = jittered_sampling

#         # Start the JVM if not already running
#         if not isJVMStarted():
#             startJVM(getDefaultJVMPath(), "-ea", f"-Djava.class.path={jar_location}")

#         # Import the TE class from JIDT
#         self.teCalcClass = JPackage("infodynamics.measures.spiking.integration").TransferEntropyCalculatorSpikingIntegration

#     def compute_TE(self, source_events, destination_events):
#         """
#         Compute the Transfer Entropy between two spike trains.
        
#         Args:
#             source_events (np.array): Array of spike times for the source process.
#             destination_events (np.array): Array of spike times for the destination process.
        
#         Returns:
#             result (float): Computed Transfer Entropy in nats.
#         """
#         # Create an instance of the TE calculator
#         teCalc = self.teCalcClass()
        
#         # Set properties
#         teCalc.setProperty("knns", str(self.knns))
#         teCalc.setProperty("DEST_PAST_INTERVALS", ",".join(map(str, self.dest_past_intervals)))
#         teCalc.setProperty("SOURCE_PAST_INTERVALS", ",".join(map(str, self.source_past_intervals)))
#         teCalc.setProperty("DO_JITTERED_SAMPLING", "true" if self.jittered_sampling else "false")
        
#         # Start adding observations
#         teCalc.startAddObservations()
        
#         # Convert spike trains into Java arrays
#         source_array = JArray(JDouble, 1)(source_events)
#         dest_array = JArray(JDouble, 1)(destination_events)
        
#         # Add observations
#         teCalc.addObservations(source_array, dest_array)
        
#         # Finalize the observation process
#         teCalc.finaliseAddObservations()
        
#         # Compute the average local Transfer Entropy
#         result = teCalc.computeAverageLocalOfObservations()
        
#         return result

#     def compute_TE_moving_window(self, source_events, destination_events, window_size, step_size):
#         """
#         Compute the Transfer Entropy dynamics using a moving window.
        
#         Args:
#             source_events (np.array): Array of spike times for the source process.
#             destination_events (np.array): Array of spike times for the destination process.
#             window_size (float): Size of the moving window (time units).
#             step_size (float): Step size for moving the window forward (time units).
        
#         Returns:
#             time_points (list): List of window midpoints where TE was computed.
#             te_dynamics (list): List of computed TE values for each window.
#         """
#         te_values = []  # To store TE values for each window
#         time_points = []  # To store the center of each window
        
#         # Determine the min and max time from the events
#         min_time = min(np.min(source_events), np.min(destination_events))
#         max_time = max(np.max(source_events), np.max(destination_events))
        
#         # Define the start and end time for the first window
#         current_start = min_time
#         current_end = current_start + window_size
        
#         while current_end <= max_time:
#             # Get events in the current window for both source and destination
#             source_window = source_events[(source_events >= current_start) & (source_events < current_end)]
#             dest_window = destination_events[(destination_events >= current_start) & (destination_events < current_end)]
            
#             if len(source_window) > 0 and len(dest_window) > 0:
#                 # Compute TE for the current window
#                 te_value = self.compute_TE(source_window, dest_window)
#                 te_values.append(te_value)
#             else:
#                 # If there are no events in the window, append NaN
#                 te_values.append(np.nan)
            
#             # Append the midpoint of the current window to time_points
#             time_points.append((current_start + current_end) / 2)
            
#             # Move the window forward by the step size
#             current_start += step_size
#             current_end += step_size
        
#         return time_points, te_values

#     def shutdown(self):
#         """
#         Shutdown the JVM after computations are done.
#         """
#         if isJVMStarted():
#             shutdownJVM()


# def test_transfer_entropy_spiking():
#     # Generate two sets of events (spike trains) for testing
#     NUM_SPIKES = 1000
#     MAX_TIME = 100.0  # Spike trains are generated within this time window
    
#     def generate_spike_train(num_spikes, max_time):
#         spike_train = np.random.uniform(0, max_time, num_spikes)
#         spike_train.sort()  # Sort the spikes in ascending order
#         return spike_train
    
#     source_spike_train = generate_spike_train(NUM_SPIKES, MAX_TIME)
#     destination_spike_train = generate_spike_train(NUM_SPIKES, MAX_TIME)

#     # Initialize the TE calculator class
#     te_calculator = TransferEntropyCalculator_spiking(
#         jar_location=jar_location, 
#         knns=4, 
#         dest_past_intervals=[1, 2], 
#         source_past_intervals=[1, 2], 
#         jittered_sampling=False
#     )

#     # Compute regular TE
#     te_value = te_calculator.compute_TE(source_spike_train, destination_spike_train)
#     print(f"Computed TE: {te_value:.4f} nats")

#     # Moving window TE computation
#     window_size = 10.0  # Window size for moving window TE
#     step_size = 5.0  # Step size for moving window TE
#     time_points, te_dynamics = te_calculator.compute_TE_moving_window(
#         source_spike_train, destination_spike_train, window_size, step_size
#     )

#     # Print TE dynamics
#     for t, te in zip(time_points, te_dynamics):
#         print(f"Time {t:.2f}: TE = {te:.4f}")

#     # Shutdown the JVM when done
#     te_calculator.shutdown()



class TransferEntropyCalculator_continuous:
    def __init__(self, jar_location=None, normalise=True, kernel_width=0.5):
        if jar_location is None:
            jar_location = jitd_jar_location_default
        if not isJVMStarted():
            startJVM(getDefaultJVMPath(), "-ea", f"-Djava.class.path={jar_location}")
        self.teCalcClass = JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
        self.normalise = normalise
        self.kernel_width = kernel_width

    def compute_TE(self, source_array, destination_array, history_length=1):
        teCalc = self.teCalcClass()
        teCalc.setProperty("NORMALISE", "true" if self.normalise else "false")
        teCalc.initialise(history_length, self.kernel_width)
        teCalc.setObservations(JArray(JDouble, 1)(source_array), JArray(JDouble, 1)(destination_array))
        return teCalc.computeAverageLocalOfObservations()

    def shutdown(self):
        if isJVMStarted():
            shutdownJVM()


def test_transfer_entropy_continous():
    import random
    import math

    numObservations = 1000
    covariance = 0.4

    # Random normal source
    sourceArray = [random.normalvariate(0,1) for _ in range(numObservations)]
    # Partially correlated destination
    destArray = [0] + [
        sum(pair) for pair in zip(
            [covariance * y for y in sourceArray[:-1]],
            [(1 - covariance) * y for y in [random.normalvariate(0,1) for _ in range(numObservations - 1)]]
        )
    ]

    cont_te_calc = TransferEntropyCalculator_continuous(normalise=True, kernel_width=0.5)
    te_value = cont_te_calc.compute_TE(sourceArray, destArray, history_length=1)
    print(f"Continuous TE: {te_value:.4f} bits")

    cont_te_calc.shutdown()

# Example usage
if __name__ == "__main__":
    # test_transfer_entropy_spiking()
    test_tra
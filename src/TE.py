import os
import numpy as np
from abc import ABC, abstractmethod
from jpype import *
import tqdm

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

        self.teCalcClass = self._define_teCalcClass()
        
    @abstractmethod
    def _define_teCalcClass(self):
        '''
        Define the TE calculator class
        Example:
        return JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
        '''
        raise NotImplementedError("This method should be implemented by subclasses")
    
    @abstractmethod
    def compute_TE(self, source_array, destination_array):
        pass
    
    def compute_TE_moving_window(self, source_array, destination_array, window_size, step_size, **kwargs):
        te_values = []
        window_centers = []
        for start in tqdm.tqdm(range(0, len(source_array) - window_size + 1, step_size), desc="Computing moving window"):
            end = start + window_size
            te_value = self.compute_TE(source_array[start:end], destination_array[start:end], **kwargs)
            te_values.append(te_value)
            window_centers.append(start + window_size // 2)
        return window_centers, te_values

    
    def shutdown(self):
        if isJVMStarted():
            shutdownJVM()
    

class TransferEntropyCalculator_continuous_kernel(TransferEntropyCalculator):
    def _define_teCalcClass(self):
        return JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
    
    def compute_TE(self, source_array, destination_array, normalise=True, kernel_width=0.5, history_length=1):
        teCalc = self.teCalcClass()
        teCalc.setProperty("NORMALISE", "true" if normalise else "false")
        teCalc.initialise(history_length, kernel_width)
        teCalc.setObservations(JArray(JDouble, 1)(source_array), JArray(JDouble, 1)(destination_array))
        return teCalc.computeAverageLocalOfObservations()

    
class TransferEntropyCalculator_continuous_kraskov(TransferEntropyCalculator):
    def _define_teCalcClass(self):
        return JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
    
    def compute_TE(self, source_array, destination_array, isPrintEstimation=False):
        teCalcClass = self.teCalcClass
        teCalc = self.teCalcClass()
        
        # Next, embed the destination only using the Ragwitz criteria:
        teCalc.setProperty(teCalcClass.PROP_AUTO_EMBED_METHOD,
                teCalcClass.AUTO_EMBED_METHOD_RAGWITZ)
        teCalc.setProperty(teCalcClass.PROP_K_SEARCH_MAX, '6') 
        teCalc.setProperty(teCalcClass.PROP_TAU_SEARCH_MAX, '6')
        # teCalc.setProperty(teCalcClass.L_PROP_NAME, "1") 
        # teCalc.setProperty(teCalcClass.L_TAU_PROP_NAME, "1")
        teCalc.initialise()
        
        # teCalc.setObservations(JArray(JDouble, 1)(source_array), JArray(JDouble, 1)(destination_array))
        teCalc.setObservations(source_array, destination_array)
        
        if isPrintEstimation:
            optimisedK = teCalc.getProperty(teCalcClass.K_PROP_NAME)
            optimisedKTau = teCalc.getProperty(teCalcClass.K_TAU_PROP_NAME)
            optimisedL = teCalc.getProperty(teCalcClass.L_PROP_NAME)
            optimisedLTau = teCalc.getProperty(teCalcClass.L_TAU_PROP_NAME)
            print(f"TE was optimised via Ragwitz criteria: k={optimisedK}, k_tau={optimisedKTau}, l={optimisedL}, l_tau={optimisedLTau}")
        return teCalc.computeAverageLocalOfObservations()

# Example usage
if __name__ == "__main__":
    # # test TransferEntropyCalculator_continuous_kernel
    # te_calculator = TransferEntropyCalculator_continuous_kernel()
    # source_array = np.random.rand(100)
    # destination_array = np.random.rand(100)
    # te_value = te_calculator.compute_TE(source_array, destination_array)
    # print(f"Transfer Entropy: {te_value:.4f} nats")
    
    # # test compute_TE_moving_window
    # time_points, te_values = te_calculator.compute_TE_moving_window(source_array, destination_array, window_size=10, step_size=5)
    # print("Time points:", time_points)
    # print("TE values:", te_values)
    
    # te_calculator.shutdown()
    
    
    # test TransferEntropyCalculator_continuous_kraskov
    te_calculator = TransferEntropyCalculator_continuous_kraskov()
    source_array = np.random.rand(1000)
    destination_array = np.random.rand(1000)
    te_value = te_calculator.compute_TE(source_array, destination_array)
    print(f"Transfer Entropy: {te_value:.4f} nats")
    
    # test compute_TE_moving_window 
    time_points, te_values = te_calculator.compute_TE_moving_window(source_array, destination_array, window_size=100, step_size=20)
    print("Time points:", time_points)
    print("TE values:", te_values)
    
    te_calculator.shutdown()
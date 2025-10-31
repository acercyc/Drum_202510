% Add JIDT jar library to the path, and disable warnings that it's already there:
warning('off','MATLAB:Java:DuplicateClass');
javaaddpath('C:\\Project\\infodynamics-dist-1.6.1\\infodynamics.jar');
% Add utilities to the path
addpath('C:\\Project\\infodynamics-dist-1.6.1\\demos\\octave');

% 0. Load/prepare the data:
data = load('D:\\SynologyDrive\\Drive-Acer\\DeepWen\\deepwen\\home\\acercyc\\Projects\\Drum\\data\\H_240702_LED2_ewma_binary.csv');
% Column indices start from 1 in Matlab:
source = octaveToJavaDoubleArray(data(:,1));
destination = octaveToJavaDoubleArray(data(:,2));

% 1. Construct the calculator:
calc = javaObject('infodynamics.measures.continuous.kraskov.TransferEntropyCalculatorKraskov');
% 2. Set any properties to non-default values:
% No properties were set to non-default values
% 3. Initialise the calculator for (re-)use:
calc.initialise();
% 4. Supply the sample data:
calc.setObservations(source, destination);
% 5. Compute the estimate:
result = calc.computeAverageLocalOfObservations();

fprintf('TE_Kraskov (KSG)(col_0 -> col_1) = %.4f nats\n', ...
	result);

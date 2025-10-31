package infodynamics.demos.autoanalysis;

import infodynamics.utils.ArrayFileReader;
import infodynamics.utils.MatrixUtils;

import infodynamics.measures.continuous.*;
import infodynamics.measures.continuous.kraskov.*;

public class GeneratedCalculator {

  public static void main(String[] args) throws Exception {

    // 0. Load/prepare the data:
    String dataFile = "D:\\SynologyDrive\\Drive-Acer\\DeepWen\\deepwen\\home\\acercyc\\Projects\\Drum\\data\\H_240702_LED2_ewma_binary.csv";
    ArrayFileReader afr = new ArrayFileReader(dataFile);
    double[][] data = afr.getDouble2DMatrix();
    double[] source = MatrixUtils.selectColumn(data, 0);
    double[] destination = MatrixUtils.selectColumn(data, 1);

    // 1. Construct the calculator:
    TransferEntropyCalculatorKraskov calc;
    calc = new TransferEntropyCalculatorKraskov();
    // 2. Set any properties to non-default values:
    // No properties were set to non-default values
    // 3. Initialise the calculator for (re-)use:
    calc.initialise();
    // 4. Supply the sample data:
    calc.setObservations(source, destination);
    // 5. Compute the estimate:
    double result = calc.computeAverageLocalOfObservations();

    System.out.printf("TE_Kraskov (KSG)(col_0 -> col_1) = %.4f nats\n",
        result);
  }
}


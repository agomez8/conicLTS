/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package data;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Class with data for a least square model |y-Ax|_2^2.
 *
 * @author AGomez
 */
public class Model_Real extends Regression {

    //--------------------------------------------------------------------------
    // Constants
    //--------------------------------------------------------------------------
    static final double EPSILON = 1e-4;

    //--------------------------------------------------------------------------
    // Attributes
    //--------------------------------------------------------------------------
    /**
     * Dimensions.
     */
    public int totalM;

    public String[] names;

    /**
     * List with validation and test set for the response variable.
     */
    List<Double> yValidation, yTest;


    /**
     * Validation and test set for predictor variables.
     */
    List<double[]> AValidation, ATest;

    /**
     * Random number generator.
     */
    Random random;

    //--------------------------------------------------------------------------
    // Constructor
    //--------------------------------------------------------------------------
    /**
     * Constructs a linear regression model from a file. <br>
     *
     * @param name Name of the dataset to use. <br>
     * @param trainingWeight Weight of the data to be used for training. <br>
     * @param validationWeight Weight of the data to be used for validation.
     * <br>
     * @param testWeight Weight of the data to be used for testing. <br>
     * @param seed Seed for the random number generator.
     */
    public Model_Real(String name, double trainingWeight, double validationWeight, double testWeight, long seed) {
        random = new Random(seed);
        buildModel(name, trainingWeight, validationWeight, testWeight);
        standardize();
        double[] beta=getOLS();
        System.out.println("OLS:");
        for (double d : beta) {
            System.out.print(d+"");
        }
        System.out.println("");
    }

    //--------------------------------------------------------------------------
    // Getters and Setters
    //--------------------------------------------------------------------------
    //--------------------------------------------------------------------------
    // Methods
    //--------------------------------------------------------------------------
    /**
     * Builds the model from data. <br>
     *
     * @param fileName File to read from.
     */
    void buildModel(String fileName, double trainingWeight, double validationWeight, double testWeight) {
        BufferedReader br = null;
        String cvsSplitBy = ",";
        List<Double> ys = new ArrayList<>();
        List<double[]> As = new ArrayList<>();
        String line;

        try {
            br = new BufferedReader(new FileReader(fileName));
            line = br.readLine();
            names = line.split(cvsSplitBy);// First line contains headers, is ignored.
            while ((line = br.readLine()) != null) {
                String[] data = line.split(cvsSplitBy);
                n = data.length - 1;
                double[] row = new double[n];
                for (int i = 0; i < n; i++) {
                    row[i] = Double.parseDouble(data[i]);

                }
                ys.add(Double.parseDouble(data[n]));
                As.add(row);
            }

        } catch (FileNotFoundException e) {
        } catch (IOException e) {
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                }
            }
        }
        totalM = ys.size();
        double propTraining = trainingWeight / (trainingWeight + validationWeight + testWeight),
                propValidation = validationWeight / (trainingWeight + validationWeight + testWeight);

        List<Integer> classes = new ArrayList<>();
        int index = 0;
        while (index < totalM) {
            if (index <= (int) (propTraining * totalM)) {
                classes.add(0);
            } else if (index <= (int) ((propTraining + propValidation) * totalM)) {
                classes.add(1);
            } else {
                classes.add(2);
            }
            index++;
        }
        Collections.shuffle(classes, random);

        List<double[]> AList = new ArrayList<>();
        AValidation = new ArrayList<>();
        ATest = new ArrayList<>();

        List<Double> yList = new ArrayList<>();
        yValidation = new ArrayList<>();
        yTest = new ArrayList<>();

        for (int i = 0; i < totalM; i++) {
            switch (classes.get(i)) {
                case 0:
                    AList.add(As.get(i));
                    yList.add(ys.get(i));
                    break;
                case 1:
                    AValidation.add(As.get(i));
                    yValidation.add(ys.get(i));
                    break;
                case 2:
                    ATest.add(As.get(i));
                    yTest.add(ys.get(i));
                    break;
            }

        }
        m = yList.size();
        y = new double[m];
        A = new double[m][n];
        for (int i = 0; i < m; i++) {
            y[i] = yList.get(i);
            A[i] = AList.get(i);
        }
    }

    @Override
    public double[][] getQPMatrix() {
        double[][] matrix = new double[n][n];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                for (int k = 0; k < m; k++) {
                    matrix[i][j] += A[k][i] * A[k][j];
                }

            }
        }
        return matrix;
    }

    @Override
    public void standardize() {
        for (int j = 0; j < n; j++) {
            // 0 mean
            double mean = 0;
            for (int i = 0; i < m; i++) {
                mean += A[i][j];
            }
            mean /= (double) m;

            for (int i = 0; i < m; i++) {
                A[i][j] -= mean;
            }
            for (double[] row : AValidation) {

                row[j] -= mean;

            }
            for (double[] row : ATest) {
                row[j] -= mean;
            }

            // 1 norm
            double var = 0;
            for (int i = 0; i < m; i++) {
                var += A[i][j] * A[i][j];

            }
            for (int i = 0; i < m; i++) {
                A[i][j] /= Math.sqrt(var);
            }
            for (double[] row : AValidation) {
                row[j] /= Math.sqrt(var);
            }
            for (double[] row : ATest) {
                row[j] /= Math.sqrt(var);
            }
        }

        double mean = 0;
        for (int i = 0; i < m; i++) {
            mean += y[i];

        }
        mean /= (double) m;

        for (int i = 0; i < m; i++) {
            y[i] -= mean;

        }
        for (int i = 0; i < yValidation.size(); i++) {
            yValidation.set(i, yValidation.get(i) - mean);
        }
        for (int i = 0; i < yTest.size(); i++) {
            yTest.set(i, yTest.get(i) - mean);
        }

        // 1 norm
        double var = 0;
        for (int i = 0; i < m; i++) {
            var += y[i] * y[i];
        }
        for (int i = 0; i < m; i++) {
            y[i] /= Math.sqrt(var);
        }
        for (int i = 0; i < yValidation.size(); i++) {
            yValidation.set(i, yValidation.get(i) / Math.sqrt(var));
        }
        for (int i = 0; i < yTest.size(); i++) {
            yTest.set(i, yTest.get(i) / Math.sqrt(var));
        }
    }

    @Override
    public double[] computeMetrics(double[] beta) {
        double[] metrics = new double[7];

        // Metric 0, 1 Training error and training R^2.
        double training = 0, validation = 0, test = 0;
        double trainingVal, validationVal, testVal;
        double var = 0, mean = 0;
        for (int i = 0; i < m; i++) {
            trainingVal = y[i];
            mean += y[i];
            for (int j = 0; j < n; j++) {
                trainingVal -= A[i][j] * beta[j];
            }
            training += trainingVal * trainingVal;
        }
        mean /= (double) m;
        for (int i = 0; i < m; i++) {
            var += (y[i] - mean) * (y[i] - mean);
        }
        metrics[0] = training;
        metrics[1] = 1 - training / var;

        // Metric 2, 3 Validation error and validation R^2.
        var = 0;
        mean = 0;
        for (int i = 0; i < yValidation.size(); i++) {
            validationVal = yValidation.get(i);
            mean += yValidation.get(i);
            for (int j = 0; j < n; j++) {
                validationVal -= AValidation.get(i)[j] * beta[j];
            }
            validation += validationVal * validationVal;
        }
        mean /= (double) yValidation.size();
        for (int i = 0; i < yValidation.size(); i++) {
            var += (yValidation.get(i) - mean) * (yValidation.get(i) - mean);
        }
        metrics[2] = validation;
        metrics[3] = 1 - validation / var;

        // Metric 4, 5 Test error and test R^2.
        var = 0;
        mean = 0;
        for (int i = 0; i < yTest.size(); i++) {
            testVal = yTest.get(i);
            mean += yTest.get(i);
            for (int j = 0; j < n; j++) {
                testVal -= ATest.get(i)[j] * beta[j];
            }
            test += testVal * testVal;
        }
        mean /= (double) yTest.size();
        for (int i = 0; i < yTest.size(); i++) {
            var += (yTest.get(i) - mean) * (yTest.get(i) - mean);
        }
        metrics[4] = test;
        metrics[5] = 1 - test / var;

        // 6: Sparsity
        int nonzeros = 0;

        for (int i = 0; i < n; i++) {
            if (Math.abs(beta[i]) > EPSILON) {
                nonzeros++;
            }
        }
        metrics[6] = nonzeros;

        return metrics;
    }

    @Override
    public int getN() {
        return n;
    }

    @Override
    public int getM() {
        return m;
    }

    @Override
    public double[] getY() {
        return y;
    }

    @Override
    public double[][] getA() {
        return A;
    }

    @Override
    public double getXtY() {
        double max = 0, val;
        for (int i = 0; i < n; i++) {
            val = 0;
            for (int j = 0; j < m; j++) {
                val += A[j][i] * y[j];
            }
            max = Math.max(val, max);
        }

        return max;
    }

    @Override
    public double getAvgDiag() {
        double[][] R = getQPMatrix();
        double avg = 0.0;
        for (int i = 0; i < R.length; i++) {
            avg += R[i][i];
        }
        return avg /= (double) R.length;
    }

    @Override
    public void mergeTrainingValidation() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    /**
     * Computes the adjusted R^2 value. <br>
     *
     * @param residuals The residuals. <br>
     * @param features The number of features. <br>
     * @return The adjusted R^2 value.
     */
    double computeAdjustedR2(double[] residuals, int features
    ) {
        double SSresiduals = 0;
        for (double residual : residuals) {
            SSresiduals += residual * residual;
        }
//        double R2 = 1 - SSresiduals / totalVariance;
        double R2 = 1 - SSresiduals;
        return 1 - (1 - R2) * (n - 1) / (n - features);
    }

    /**
     * Computes the Bayesian information criterion. <br>
     *
     * @param residuals The residuals. <br>
     * @param features The number of features. <br>
     * @return The Bayesian information criterion.
     */
    double computeBIC(double[] residuals, int features
    ) {
        double SSresiduals = 0;
        for (double residual : residuals) {
            SSresiduals += residual * residual;
        }
        return n * Math.log(SSresiduals / (double) n) + Math.log(n) * features;
    }

    /**
     * Computes the Akaike information criterion. <br>
     *
     * @param residuals The residuals. <br>
     * @param features The number of features. <br>
     * @return The Akaike information criterion.
     */
    double computeAIC(double[] residuals, int features
    ) {
        double SSresiduals = 0;
        for (double residual : residuals) {
            SSresiduals += residual * residual;
        }
        return n * Math.log(SSresiduals / (double) n) + 2 * features;
    }

    /**
     * Computes the corrected Akaike information criterion. <br>
     *
     * @param residuals The residuals. <br>
     * @param features The number of features. <br>
     * @return The corrected Akaike information criterion.
     */
    double computeCorrectedAIC(double[] residuals, int features
    ) {
        double SSresiduals = 0;
        for (double residual : residuals) {
            SSresiduals += residual * residual;
        }
        return n * Math.log(SSresiduals / (double) n) + n * (n + features) / (double) (n - features - 2);
    }

    @Override
    public double[] computeCriteria(double[] z, double[] residuals) {
        int features =0;
        for (double zi : z) {
            features+= zi>0.5?1:0;
        }
        
        return new double[]{computeAdjustedR2(residuals, features),
            computeBIC(residuals, features),computeCorrectedAIC(residuals, features)};
    }

    
    

}

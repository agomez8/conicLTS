/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.stat.descriptive.rank.Median;

/**
 * Class with data for a least square model |y-Ax|_2^2.
 *
 * @author AGomez
 */
public class Model_SyntLS extends Regression {

    //--------------------------------------------------------------------------
    // Constants
    //--------------------------------------------------------------------------
    static final double EPSILON = 1e-4;

    static enum Generation {
        independentMatrix, independentResponse,
        correlatedMatrix, correlatedResponse
    }

    //--------------------------------------------------------------------------
    // Attributes
    //--------------------------------------------------------------------------
    /**
     * Response variable, m-dimensional vector.
     */
    public double[] yValidation;

    /**
     * True value of the regression coefficients.
     */
    public double[] trueBeta, meansA, varsA;
    
    public double meanY,varY;

    /**
     * Model matrix, mxn dimensional matrix.
     */
    public double[][] AValidation;

    /**
     * Covariance for rows of A
     */
    public double[][] Sigma;

    /**
     * Error variance.
     */
    public double sigma2;

    public boolean intercept;

    /**
     * Random number generator.
     */
    RandomGenerator random;

    //--------------------------------------------------------------------------
    // Constructor
    //--------------------------------------------------------------------------
    /**
     * Constructs a linear regression model from a file. <br>
     *
     * @param n Number of predictors. <br>
     * @param m Number of observations. <br>
     * @param k True sparsity. <br>
     * @param pattern Sparsity pattern. <br>
     * @param autocorrelation Predictor autocorrelation level. <br>
     * @param snr Signal-to-Noise Ratio. <br>
     * @param seed Seed for the random number generator.
     */
    public Model_SyntLS(int n, int m, int k, int pattern, double autocorrelation, double snr, int seed) {

        this.n = n;
        this.m = m;
        random = new JDKRandomGenerator(seed);
        trueBeta = generateBeta(k, pattern <= 2 ? 2 : 1);
//        generateA(autocorrelation);
//        generateY(snr);
        this.intercept = true;
        generateAOutlier(pattern <= 2 ? 0 : 0.5);
//        standardizeA();
        generateYOutlier();
//        this.intercept = (pattern%2)==1;

        perturbData(autocorrelation, pattern % 2);

//        mergeTrainingSet();
        standardize();
//        standardizeY();

//        for (int i = 0; i < A.length; i++) {
//            for (int j = 0; j < A[i].length; j++) {
//                System.out.print(A[i][j]+" ");
//            }
//            System.out.println(y[i]);
//        }
//        generateY(snr);
    }

    //--------------------------------------------------------------------------
    // Getters and Setters
    //--------------------------------------------------------------------------
    //--------------------------------------------------------------------------
    // Methods
    //-------------------------------------------------------------------------- 
    /**
     * Generates beta according to a given sparsity and pattern. <br>
     *
     * @param k The number of nonzeros in beta. <br>
     * @param pattern The sparsity pattern. <br>
     * @return beta.
     */
    private double[] generateBeta(int k, int pattern) {
        double[] beta = new double[n];
        switch (pattern) {
            case 1:
//                int period = n / k;
                for (int i = 0; i < k; i++) {
                    beta[i] = -1 + 2 * random.nextDouble();
//                    beta[period * i] = 1;
//                    beta[period * i] = random.nextBoolean() ? 1 : -1;
                }
                break;
            case 2:
                for (int i = 0; i < k; i++) {
                    beta[i] = 1;
//                    beta[i]=random.nextDouble();
                }
                break;
            case 3:
                double distance = 9.5 / (double) (k - 1);
                for (int i = 0; i < k; i++) {
                    beta[i] = 10 - distance * i;
                }
                break;
            case 5:
                for (int i = 0; i < k; i++) {
                    beta[i] = 1;
                }
                for (int i = k; i < n; i++) {
                    beta[i] = Math.pow(0.5, i + 1 - k);
                }
                break;
        }
        return beta;
    }

    /**
     * Generates the model matrix A and A validation. <br>
     *
     * @param autocorrelation The autocorrelation used to generate A. <br>
     */
    private void generateA(double autocorrelation) {
        double[] means = new double[n];
        Sigma = new double[n][n];
        for (int i = 0; i < n; i++) {
            Sigma[i][i] = 1;
            for (int j = i + 1; j < n; j++) {
                Sigma[i][j] = Math.pow(autocorrelation, j - i);
                Sigma[j][i] = Sigma[i][j];
            }
        }
        MultivariateNormalDistribution distribution = new MultivariateNormalDistribution(random, means, Sigma);

        A = distribution.sample(m);
        AValidation = distribution.sample(m);

        double row = 0, max = Double.NEGATIVE_INFINITY, lambda;
        ddLambda = 0;
        for (int i = 0; i < A.length; i++) {
            row = 0;
//            lambda=2*A[i][i];
            for (int j = 0; j < A[i].length; j++) {
                row += A[i][j] * A[i][j];
//                lambda-=Math.abs(A[i][i]);
            }
            max = Math.max(row, max);
//            ddLambda=Math.max(ddLambda, -lambda);

        }
//        System.out.println("Max row size: "+max);
        normA = max;
    }

    /**
     * Generates the model matrix A and A validation. <br>
     *
     * @param autocorrelation The autocorrelation used to generate A. <br>
     */
    private void generateAOutlier(double autocorrelation) {
        double[] means = new double[n];
        Sigma = new double[n][n];
        for (int i = 0; i < n; i++) {
            Sigma[i][i] = 100;
            for (int j = i + 1; j < n; j++) {
                Sigma[i][j] = Math.pow(autocorrelation, j - i);
                Sigma[j][i] = Sigma[i][j];
            }
        }
        MultivariateNormalDistribution distribution = new MultivariateNormalDistribution(random, means, Sigma);

        A = distribution.sample(m);
        if (intercept == true) {
            for (int i = 0; i < m; i++) {
                A[i][n - 1] = 1;
            }
        }

        AValidation = distribution.sample(m);

        double row = 0, max = Double.NEGATIVE_INFINITY, lambda;
        ddLambda = 0;
        for (int i = 0; i < A.length; i++) {
            row = 0;
//            lambda=2*A[i][i];
            for (int j = 0; j < A[i].length; j++) {
                row += A[i][j] * A[i][j];
//                lambda-=Math.abs(A[i][i]);
            }
            max = Math.max(row, max);
//            ddLambda=Math.max(ddLambda, -lambda);

        }
//        System.out.println("Max row size: "+max);
        normA = max;
    }

    /**
     * Generates the response vectors y and yValidation. <br>
     *
     * @param snr The desired signal-to-noise ratio. <br>
     * @param beta The true regression coefficients.
     */
    private void generateY(double snr) {
        double[] means = new double[m];
        double[] meansValidation = new double[m];
        double[][] covariance = new double[m][m];
        double variance = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                variance += Sigma[i][j] * trueBeta[i] * trueBeta[j];
            }
        }
        sigma2 = variance / snr;

        for (int i = 0; i < m; i++) {
            covariance[i][i] = sigma2;
            for (int j = 0; j < n; j++) {
                means[i] += A[i][j] * trueBeta[j];
                meansValidation[i] += AValidation[i][j] * trueBeta[j];
            }
        }
        MultivariateNormalDistribution distribution = new MultivariateNormalDistribution(random, means, covariance);
        y = distribution.sample();

        MultivariateNormalDistribution distributionValidation = new MultivariateNormalDistribution(random, meansValidation, covariance);
        yValidation = distributionValidation.sample();
    }

    /**
     * Generates the response vectors y and yValidation. <br>
     *
     * @param snr The desired signal-to-noise ratio. <br>
     * @param beta The true regression coefficients.
     */
    private void generateYOutlier() {
        double[] means = new double[m];
        double[] meansValidation = new double[m];
        double[][] covariance = new double[m][m];
        double variance = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                variance += Sigma[i][j] * trueBeta[i] * trueBeta[j];
            }
        }
        sigma2 = 10;

        for (int i = 0; i < m; i++) {
            covariance[i][i] = sigma2;
            for (int j = 0; j < n; j++) {
                means[i] += A[i][j] * trueBeta[j];
                meansValidation[i] += AValidation[i][j] * trueBeta[j];
            }
        }
        MultivariateNormalDistribution distribution = new MultivariateNormalDistribution(random, means, covariance);
        y = distribution.sample();

        MultivariateNormalDistribution distributionValidation = new MultivariateNormalDistribution(random, meansValidation, covariance);
        yValidation = distributionValidation.sample();
    }

    /**
     * Generates the response vectors y and yValidation. <br>
     *
     * @param snr The desired signal-to-noise ratio. <br>
     * @param beta The true regression coefficients.
     */
    private void perturbData(double proportion, int type) {
        switch (type) {
            case 0:
                for (int i = 0; i < m * proportion; i++) {
                    A[i][0] += 1000;
//                      y[i]+=1000;
                }
                break;
            case 1:
                for (int i = 0; i < m * proportion; i++) {
                    y[i] += 1000;
                }
                break;
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
        double constant = 0;
        meansA = new double[n]; 
        varsA = new double[n];
        for (int j = 0; j < n; j++) {
            // 0 mean

            double meanValidation = 0;

            for (int i = 0; i < m; i++) {

                meansA[j] += A[i][j];
                meanValidation += AValidation[i][j];
            }

            meansA[j] /= (double) m;
            meanValidation /= (double) m;
            if (!intercept || j != n - 1) {
                for (int i = 0; i < m; i++) {
                    A[i][j] -= meansA[j];
                    AValidation[i][j] -= meanValidation;
                }
            }

            // 1 norm
            double varValidation = 0;
            for (int i = 0; i < m; i++) {
                varsA[j] += A[i][j] * A[i][j];
                varValidation += AValidation[i][j] * AValidation[i][j];
            }
            for (int i = 0; i < m; i++) {
                A[i][j] /= Math.sqrt(varsA[j]);
                AValidation[i][j] /= Math.sqrt(varValidation);
            }

        }

        meanY = 0;
        double meanValidationY = 0;
        for (int i = 0; i < m; i++) {
            meanY += y[i];
            meanValidationY += yValidation[i];
        }
        meanY /= (double) m;
//        Median median= new Median();
//        mean=median.evaluate(y);
        meanValidationY /= (double) m;
        for (int i = 0; i < m; i++) {
            y[i] -= meanY;
            yValidation[i] -= meanValidationY;
        }

        // 1 norm
        varY = 0;
        double varValidationY = 0;
        for (int i = 0; i < m; i++) {
//            var += (y[i]-mean) * (y[i]-mean);
            varY += y[i] * y[i];
            varValidationY += yValidation[i] * yValidation[i];
        }
        for (int i = 0; i < m; i++) {
            y[i] /= Math.sqrt(varY);
            yValidation[i] /= Math.sqrt(varValidationY);
        }

//        if (intercept) {
//            trueBeta[n - 1] /= Math.sqrt(varY);
//            trueBeta[n - 1] -= meanY / Math.sqrt(varY);
//            for (int j = 0; j < n - 1; j++) {
//                trueBeta[n - 1] += trueBeta[j] * mean[j] / Math.sqrt(varY);
//                trueBeta[j] *= Math.sqrt(var[j] / varY);
//            }
//            trueBeta[n - 1] *= Math.sqrt(var[n - 1]);
//        }
    }

    public void standardizeY() {

        double mean = 0, meanValidation = 0;
        for (int i = 0; i < m; i++) {
            mean += y[i];
            meanValidation += yValidation[i];
        }
        mean /= (double) m;
//        Median median= new Median();
//        mean=median.evaluate(y);
        meanValidation /= (double) m;
        for (int i = 0; i < m; i++) {
            y[i] -= mean;
            yValidation[i] -= meanValidation;
        }

        if (this.intercept) {
            trueBeta[n - 1] = -mean;
        }

        // 1 norm
        double var = 0, varValidation = 0;
        for (int i = 0; i < m; i++) {
//            var += (y[i]-mean) * (y[i]-mean);
            var += y[i] * y[i];
            varValidation += yValidation[i] * yValidation[i];
        }
        for (int i = 0; i < m; i++) {
            y[i] /= Math.sqrt(var);
            yValidation[i] /= Math.sqrt(varValidation);
        }
        for (int i = 0; i < n; i++) {
            trueBeta[i] /= Math.sqrt(var);
        }
    }

    public void standardizeA() {
        for (int j = 0; j < n; j++) {
            // 0 mean
            double mean = 0, meanValidation = 0;
            for (int i = 0; i < m; i++) {
                mean += A[i][j];
                meanValidation += AValidation[i][j];
            }
            mean /= (double) m;
            meanValidation /= (double) m;
            for (int i = 0; i < m; i++) {
                A[i][j] -= mean;
                AValidation[i][j] -= meanValidation;
            }

            // 1 norm
            double var = 0, varValidation = 0;
            for (int i = 0; i < m; i++) {
                var += A[i][j] * A[i][j];
                varValidation += AValidation[i][j] * AValidation[i][j];
            }
            for (int i = 0; i < m; i++) {
                A[i][j] /= Math.sqrt(var);
                AValidation[i][j] /= Math.sqrt(varValidation);
            }
        }
    }
    
    /**
     * Converts back to pre-standardization values. <br>
     * @param xSol Solution found.
     * @return 
     */
    public double[] convert(double[] xSol)
    {
        int p=xSol.length;
        double [] vals=new double[p];
        vals[p-1]=xSol[p-1]*Math.sqrt(varY/varsA[p-1])+meanY;
        for (int i = 0; i < p-1; i++) {
            vals[i]=xSol[i]*Math.sqrt(varY/varsA[i]);
            vals[p-1]-=vals[i]*meansA[i];
        }
        return vals;
    }

    @Override
    public double[] computeMetrics(double[] beta) {
        double[] metrics = new double[7];

        // Metric 0 and 1: Training error and prediction error
        double training = 0, prediction = 0;
        double trainingVal, predictionVal;
        for (int i = 0; i < m; i++) {
            trainingVal = y[i];
//            System.out.println(i + " " + m);
            predictionVal = yValidation[i];
//            predictionVal = 0;
            for (int j = 0; j < n; j++) {
                trainingVal -= A[i][j] * beta[j];
                predictionVal -= AValidation[i][j] * beta[j];
            }
            training += trainingVal * trainingVal;
            prediction += predictionVal * predictionVal;
        }
        metrics[0] = training;
        metrics[1] = prediction;

        double denominator = 0, numerator = 0;
        for (int i = 0; i < n-1; i++) {
            for (int j = 0; j < n-1; j++) {
                numerator += Sigma[i][j] * (beta[i] - trueBeta[i]) * (beta[j] - trueBeta[j]);
                denominator += Sigma[i][j] * trueBeta[i] * trueBeta[j];
            }
        }

        // Metric 2: Relative risk
        metrics[2] = numerator / denominator;

        // Metric 3: Relative test error
        metrics[3] = (numerator + sigma2) / sigma2;

        // Metric 4: Proportion of variance explained
        metrics[4] = 1 - (numerator + sigma2) / (denominator + sigma2);

        // Metric 5 and 6: Sparsity
//        int nonzeros = 0, trueNonzeros = 0;
//        for (int i = 0; i < n; i++) {
//            if (Math.abs(beta[i]) > EPSILON) {
//                nonzeros++;
//                if (trueBeta[i] != 0) {
//                    trueNonzeros++;
//                }
//            }
//        }
//        metrics[5] = nonzeros;
//        metrics[6] = trueNonzeros;

        return metrics;
    }

    /**
     * Computes the number of correctly fixed variables. <br>
     *
     * @param indicators Array with the fixed variables: <br>
     * -1 if fixed to 0, 1 if fixed to 1, 0 if not fixed <br>
     * @return The number of correctly fixed variables.
     */
    public int computeMetricsFixed(int[] indicators) {

        int fixed = 0;
        for (int i = 0; i < n; i++) {
            if (indicators[i] == 1 & trueBeta[i] != 0) {
                fixed++;

            } else if (indicators[i] == -1 & trueBeta[i] == 0) {
                fixed++;
            }
        }

        return fixed;
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
        double[] newY = new double[y.length * 2];
        double[][] newA = new double[A.length * 2][A[0].length];

        for (int i = 0; i < y.length; i++) {
            newY[i] = y[i];
            newY[i + y.length] = yValidation[i];
            newA[i] = A[i];
            newA[i + y.length] = AValidation[i];

        }
        y = newY;
        A = newA;
        m = y.length;
    }

    @Override
    public double[] computeCriteria(double[] beta, double[] residuals) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    /**
     * Prints the instance to a file. Does not do anything if file exists. <br>
     *
     * @param path The path to the file to print to. <br>
     * @throws IOException
     */
    public void printInstance(String path) throws IOException {
        File file = new File(path);
        if (!file.exists()) {
            try ( FileWriter out = new FileWriter(file, true)) {
                out.write("A\n");
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        out.write(A[i][j] + ",");
                    }
                    out.write(y[i] + "\n");
                }
            }
        }
    }
}

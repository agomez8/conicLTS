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
import java.util.List;

/**
 * Class with data for a least square model |y-Ax|_2^2.
 *
 * @author AGomez
 */
public class Model_LS extends Regression{
    
    //--------------------------------------------------------------------------
    // Constants
    //--------------------------------------------------------------------------
    /**
     * Integrality precision.
     */
    static final double EPSILON = 1e-4;

    //--------------------------------------------------------------------------
    // Attributes
    //--------------------------------------------------------------------------
   

    //--------------------------------------------------------------------------
    // Constructor
    //--------------------------------------------------------------------------
    /**
     * Constructs a linear regression model from a file. <br>
     *
     * @param fileName The path to the file.
     */
    public Model_LS(String fileName) {
        buildModel(fileName);
        standardize();
//        System.out.println("OLS:");
//        double[] beta=getOLS();
//        for (double d : beta) {
//            System.out.print(d+" ");
//        }
//        System.out.println("");
        
        double row = 0, max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < A.length; i++) {
            row = 0;
            for (int j = 0; j < A[i].length; j++) {
                row += A[i][j] * A[i][j];
            }
            max = Math.max(row, max);

        }
//        System.out.println("Max row size: "+max);
        normA = max;
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
    void buildModel(String fileName) {
        BufferedReader br = null;
        String cvsSplitBy = ",";
        List<Double> ys = new ArrayList<>();
        List<double[]> As = new ArrayList<>();
        String line = "";

        try {
            br = new BufferedReader(new FileReader(fileName));
            br.readLine(); // First line contains headers, is ignored.

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
        m = ys.size();
        y = new double[m];
        A = new double[m][n];
        for (int i = 0; i < m; i++) {
            y[i] = ys.get(i);
            A[i] = As.get(i);
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

            // 1 norm
            double var = 0;
            for (int i = 0; i < m; i++) {
                var += A[i][j] * A[i][j];
            }
            if (var != 0) {

                for (int i = 0; i < m; i++) {
                    A[i][j] /= Math.sqrt(var);
//                        A[i][j] /= Math.sqrt(var/(double)(m));
                }
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

        // 1 norm
        double var = 0;
        for (int i = 0; i < m; i++) {
            var += y[i] * y[i];
        }
        for (int i = 0; i < m; i++) {
            y[i] /= Math.sqrt(var);
//            y[i] /= Math.sqrt(var/(double)(m));
        }
    }

    @Override
    public double[] computeMetrics(double[] beta) {
        double[] metrics = new double[2];
        
        // Metric 0 and 1: Training error and prediction error
        double training=0;
        double trainingVal;
        for (int i = 0; i < m; i++) {
            trainingVal=y[i];
            for (int j = 0; j < n; j++) {
                trainingVal -= A[i][j]*beta[j];
            }
            training += trainingVal * trainingVal;
            }
        metrics[0]=training;
        
       
        int nonzeros = 0;
        for (int i = 0; i < n; i++) {
            if(Math.abs(beta[i])>EPSILON)
            {
                nonzeros++;
                
            }
        }
        metrics[1] = nonzeros;
        
        
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
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public double getAvgDiag() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void mergeTrainingValidation() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    /**
     * Computes the  R^2 value. <br>
     *
     * @param residuals The residuals. <br>
     * @param features The number of features. <br>
     * @return The adjusted R^2 value.
     */
    double computeR2(double[] residuals, double features
    ) {
        double SSresiduals = 0;
        for (double residual : residuals) {
            SSresiduals += residual * residual;
        }

//        double R2 = 1 - SSresiduals / totalVariance;
        return 1 - SSresiduals;
//        System.out.println("R2 "+R2);
        

       
    }

    /**
     * Computes the adjusted R^2 value. <br>
     *
     * @param residuals The residuals. <br>
     * @param features The number of features. <br>
     * @return The adjusted R^2 value.
     */
    double computeAdjustedR2(double[] residuals, double features
    ) {
        double SSresiduals = 0;
        for (double residual : residuals) {
            SSresiduals += residual * residual;
        }

//        double R2 = 1 - SSresiduals / totalVariance;
        double R2 = 1 - SSresiduals;
//        System.out.println("R2 "+R2);
        

        return 1 - (1 - R2) * (m - 1) / (double)(m - features);
    }

    /**
     * Computes the Bayesian information criterion. <br>
     *
     * @param residuals The residuals. <br>
     * @param features The number of features. <br>
     * @return The Bayesian information criterion.
     */
    double computeBIC(double[] residuals, double features
    ) {
        double SSresiduals = 0;
        for (double residual : residuals) {
            SSresiduals += residual * residual;
        }
        
        return m * Math.log(SSresiduals / (double) m) + Math.log(m) * features;
    }

    /**
     * Computes the Akaike information criterion. <br>
     *
     * @param residuals The residuals. <br>
     * @param features The number of features. <br>
     * @return The Akaike information criterion.
     */
    double computeAIC(double[] residuals, double features
    ) {
        double SSresiduals = 0;
        for (double residual : residuals) {
            SSresiduals += residual * residual;
        }
        return m * Math.log(SSresiduals / (double) m) + 2 * features;
    }

    /**
     * Computes the corrected Akaike information criterion. <br>
     *
     * @param residuals The residuals. <br>
     * @param features The number of features. <br>
     * @return The corrected Akaike information criterion.
     */
    double computeCorrectedAIC(double[] residuals, double features
    ) {
        double SSresiduals = 0;
        for (double residual : residuals) {
            SSresiduals += residual * residual;
        }
        return m * Math.log(SSresiduals / (double) m) + m * (m + features) / (double) (m - features - 2);
    }

    @Override
    public double[] computeCriteria(double[] z, double[] residuals) {
        double features =0;
        for (double zi : z) {
            features+= zi<0.5?1:0;
//            features+= zi;
        }
        
        return new double[]{features,computeR2(residuals, features),computeAdjustedR2(residuals, features),
            computeBIC(residuals, features),computeCorrectedAIC(residuals, features)};
    }
}

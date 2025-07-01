/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package data;

/**
 *
 * @author agomez
 */
public interface IRegression {
    
    
    
    //--------------------------------------------------------------------------
    // Methods
    //--------------------------------------------------------------------------
    /**
     * Gets the number of predictors. <br>
     * @return n.
     */
    public int getN();
    
    /**
     * Gets the number of datapoints. <br>
     * @return m.
     */
    public int getM();
    
    /**
     * Gets the response vector. <br>
     * @return y.
     */
    public double[] getY();
    
    /**
     * Returns the optimal solution of the ordinary least squares problem. <br>
     * @return The OLS estimator.
     */
    public double[] getOLS();
    
        /**
     * Returns the residuals corresponding to a regression vector. <br>
     * @return The residual vector.
     */
    public double[] getResiduals(double[] x);
    
  
    
    /**
     * Gets the data matrix. <br>
     * @return A.
     */
    public double[][] getA();
    
    /**
     * Returns the matrix A^\top A.
     */
    public double[][] getAtA();
    
     /**
     * Computes the objective QP matrix associated with the regression problem.
     *
     * @return The objective matrix.
     */
    public double[][] getQPMatrix();
    
    /**
     * Standardizes the data.
     */
    public void standardize();
    
    /**
     * Computes statistical evaluation metrics for a given regression vector beta. <br>
     * @param beta The regression vector. <br>
     * @return The computed metrics.
     */
    public double[] computeMetrics(double[] beta);
    
    /**
     * Computes information criteria metrics for a given vectors of indicators and residuals. <br>
     * @param zs Indicator variables. <br>
     * @param residuals Residual vector. <br>
     * @return The computed metrics.
     */
    public double[] computeCriteria(double[] zs, double[] residuals);
    
    /**
     * Returns the infinity norm of (X^T)y for regularization. <br>
     * @return maxRegularization.
     */
    public double getXtY();
    
    /**
     * Gets the average diagonal element of the QP matrix. <br>
     * @return the average diagonal element.
     */
    public double getAvgDiag();
    
    public void mergeTrainingValidation();
}

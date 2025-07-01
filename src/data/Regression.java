/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package data;

import Jama.Matrix;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.CholeskyDecomposition;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author agome
 */
public abstract class Regression implements IRegression {
    
    
    private final double EPSILON =1e-4;

    /**
     * Dimensions.
     */
    public int n, m;

    /**
     * Response variable, m-dimensional vector.
     */
    public double[] y;

    /**
     * Model matrix, mxn dimensional matrix.
     */
    public double[][] A;

    /**
     * Norm A
     */
    public double normA;

    /**
     * Minimum regularization the makes the matrix diagonally dominant.
     */
    public double ddLambda;

    @Override
    public double[] getOLS() {
        Matrix Xm = new Matrix(A);
        Matrix Ym = new Matrix(m, 1);
        for (int i = 0; i < m; i++) {
            Ym.set(i, 0, y[i]);
        }
        Matrix Beta = Xm.solve(Ym);
        double[] beta = new double[n];
        for (int i = 0; i < n; i++) {
            beta[i] = Beta.get(i, 0);
        }
        return beta;
    }

    @Override
    public double[] getResiduals(double[] x) {
//        System.out.print("Residuals: ");
        double[] residuals = new double[m];
        for (int i = 0; i < m; i++) {
            residuals[i] = y[i];
            for (int j = 0; j < n; j++) {
                residuals[i] -= A[i][j] * x[j];
            }
//            residuals[i]= residuals[i]*residuals[i];
//            System.out.print(residuals[i]+" ");
        }
//        System.out.println("");
        return residuals;
    }

    @Override
    public double[][] getAtA() {
        double[][] A = getA();
        int n = A[0].length;
        double[][] matrix = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                matrix[i][j] = 0;
                for (int k = 0; k < A.length; k++) {
                    matrix[i][j] += A[k][i] * A[k][j];
                }
            }
        }
        return matrix;
    }

    public double[][] getOutlierFormualtion(boolean hasIntercept, double lambda,double[] constant, double[] gamma, double[] b) {
        RealMatrix matrix = new Array2DRowRealMatrix(getAtA());
        RealVector yVec= new ArrayRealVector(y);
        if (!hasIntercept) {
            matrix.setEntry(0, 0, lambda + matrix.getEntry(0, 0));
        }
        for (int i = 1; i < n; i++) {
            matrix.setEntry(i, i, lambda + matrix.getEntry(i, i));
        }

        matrix = MatrixUtils.inverse(matrix);
        RealMatrix AMatrix = new Array2DRowRealMatrix(A);
        matrix = matrix.preMultiply(AMatrix);
        matrix = matrix.multiply(AMatrix.transpose());
        RealMatrix finalM = new Array2DRowRealMatrix(m, m);
        for (int i = 0; i < m; i++) {
            finalM.setEntry(i, i, 1);
        }
        finalM = finalM.subtract(matrix);
        EigenDecomposition decomp = new EigenDecomposition(finalM);
        RealMatrix D=decomp.getD();
        
        double minEigenvalue=D.getEntry(D.getRowDimension()-1,D.getRowDimension()-1);
        
        System.out.println("Min eigenvalue "+minEigenvalue);
        System.out.println("Eigenvalues");
        for (int i = 0; i < D.getRowDimension(); i++) {
            System.out.print(D.getEntry(i, i)+" ");
//            D.setEntry(i, i, Math.sqrt(D.getEntry(i, i)));
        }
        System.out.println("");
        
        if(minEigenvalue<EPSILON)
        {
            System.out.println("Minimum eigenvalue too small to extract diagonal");
            return null;
        }
        
        gamma[0]=minEigenvalue-EPSILON;
        for (int i = 0; i < finalM.getRowDimension(); i++) {
            finalM.setEntry(i, i, finalM.getEntry(i, i)-(minEigenvalue-EPSILON));
        }
        
        decomp = new EigenDecomposition(finalM);
        RealMatrix V=decomp.getV();
        D=decomp.getD();
       
//        System.out.println("Min eigenvalue "+minEigenvalue);
//        System.out.println("Eigenvalues");
        for (int i = 0; i < D.getRowDimension(); i++) {
//            System.out.print(D.getEntry(i, i)+" ");
            D.setEntry(i, i, Math.sqrt(D.getEntry(i, i)));
        }
        System.out.println("");
        
        
        
        
        RealMatrix root = D.multiply(V.transpose());
        
        RealMatrix rootInv=MatrixUtils.inverse(root).transpose();
        
        RealVector bVec= rootInv.scalarMultiply(gamma[0]).add(root).operate(yVec);
        System.arraycopy(bVec.toArray(), 0, b, 0, bVec.getDimension());
        
        RealMatrix quad=root.preMultiply(root.transpose());
        for (int i = 0; i < quad.getRowDimension(); i++) {
            quad.setEntry(i, i, gamma[0]+quad.getEntry(i, i));
        }
        constant[0]=quad.preMultiply(yVec).dotProduct(yVec)-bVec.getNorm()*bVec.getNorm();
        
        
        
//        DecimalFormat format= new DecimalFormat("0.00");
//        for (int i = 0; i < V.getRowDimension(); i++) {
//            for (int j = 0; j < V.getColumnDimension(); j++) {
//                System.out.print(format.format(root.getEntry(i, j))+" ");
//            }
//            System.out.println("");
//        }
        
        return root.getData();
    }

    public void printOutlierData(String path, boolean hasIntercept, double l2reg) {
        double[] constant= new double[1], gamma= new double[1];
        double[] b= new double[m];
        double[][] V = getOutlierFormualtion(hasIntercept, l2reg,constant,gamma,b);
        System.out.println("Constant "+constant[0]);
        System.out.println("Gamma "+gamma[0]);
        System.out.print("b ");
        for (double d : b) {
            System.out.print(d+" ");
        }
        System.out.println("");
        
        double[] newY = new Array2DRowRealMatrix(V).operate(new ArrayRealVector(y)).toArray();
        try ( FileWriter out = new FileWriter(new File( path), false)) {
            out.write("Constant,"+constant[0]+"\n");
            out.write("Gamma,"+gamma[0]+"\n");
            out.write("b\n");
            for (int i=0; i<b.length-1;i++) {
                out.write(b[i]+",");
            }
            out.write(b[b.length-1]+"\n\n");
            
            for (int i = 0; i < m; i++) {
                out.write("obs" + i + ", ");
            }
            out.write("y\n");

            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    out.write(-V[i][j] + ", ");
                }
                out.write(newY[i] + "\n");
            }

           

        } catch (IOException ex) {
            Logger.getLogger(Regression.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

}

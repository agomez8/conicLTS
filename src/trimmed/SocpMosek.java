/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package trimmed;


import data.Regression;
import mosek.*;
import org.apache.commons.math3.linear.CholeskyDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * Runs MIO formulations with Mosek. 
 * Not reported in the paper since Gurobi was faster. 
 * Code may not be updated. <br>
 * @author Andres Gomez
 */
public class SocpMosek {

    final double EPSILON = 0;

    /**
     * Mosek environment. <br>
     */
    Env env;

    /**
     * Mosek task. <br>
     */
    Task task;

    /**
     * Instance to be solved.
     */
    Regression instance;

    /**
     * Array with position of the variables
     */
    int[] x, v, z, tau;

    /**
     * Index of last variable and constraint added to the model
     */
    int varIndex, conIndex;

    double M = 1000;

    double mLambda;

    double time;

    public boolean isInt;

    //--------------------------------------------------------------------------
    // Constructor
    //--------------------------------------------------------------------------
    /**
     * Constructor.<br>
     *
     * @param instance
     */
    public SocpMosek(Regression instance) {
        env = new Env();
        task = new Task(env, 0, 0);
        this.instance = instance;
        time = 0;
        task.putintparam(mosek.iparam.num_threads, 1);
        task.putdouparam(dparam.mio_max_time, 600);
        isInt = true;
//        
        task.set_Stream(streamtype.log, new Stream() {
            public void stream(String string) {
                System.out.println(string);
            }
        });

//        buildInstance(a, ub);
    }

    //--------------------------------------------------------------------------
    // Methods
    //--------------------------------------------------------------------------
    /**
     * Builds an instance without a point. <br>
     *
     */
    void buildMinimizationZioutas(double gamma, int k) {
//        task.dispose();
        int n = instance.n;
        int m = instance.m;
        varIndex = 0;
        conIndex = 0;
        x = new int[n];
        v = new int[m];
        z = new int[m];
        task.appendvars(3 * n + 2 * m); // Continuous + auxiliary + discrete variables
//        task.appendcons(2 * n); // Big M constraints
        task.appendcons(1); // Cardinality
        int cardConstr = conIndex;
        task.putconbound(cardConstr, boundkey.up, 0, k);
        conIndex++;

        // Adds variables
        for (int i = 0; i < n; i++) {
            x[i] = varIndex;
            task.putvarbound(x[i], boundkey.fr, -0.0, +0.0);
            task.putvarname(x[i], "x" + i);
//            varIndex++;

            task.putvarbound(varIndex + 1, boundkey.fx, 0.5, 0.5);
            task.putvarname(varIndex + 1, "const" + i);

            task.putvarbound(varIndex + 2, boundkey.lo, 0.0, +0.0);
            task.putvarname(varIndex + 2, "p" + i);
            task.putcj(varIndex + 2, gamma);

            task.appendcone(conetype.rquad, 0, new int[]{varIndex + 2, varIndex + 1, x[i]});
            varIndex += 3;

        }

        for (int i = 0; i < m; i++) {
            z[i] = varIndex;
            task.putvarbound(z[i], boundkey.ra, 0.0, 1.0);
            task.putvarname(z[i], "z" + i);
            task.putaij(cardConstr, z[i], 1);
            task.putvartype(z[i], variabletype.type_int);
            varIndex++;

            v[i] = varIndex;
            task.putvarbound(v[i], boundkey.fr, -0.0, +0.0);
            task.putvarname(v[i], "v" + i);
            varIndex++;

            task.appendcons(2); // Big M
            task.putarow(conIndex, new int[]{v[i], z[i]}, new double[]{1, -M});
            task.putconbound(conIndex, boundkey.up, -0.0, 0);
            task.putarow(conIndex + 1, new int[]{v[i], z[i]}, new double[]{-1, -M});
            task.putconbound(conIndex + 1, boundkey.up, -0.0, 0);
            conIndex += 2;
        }

        double[][] A = instance.A;

        for (int i = 0; i < A.length; i++) {
            task.appendvars(3);

            // epigraph variable
            task.putvarbound(varIndex, boundkey.lo, 0.0, +0.0);
            task.putvarname(varIndex, "s" + i);
            task.putcj(varIndex, 1);

            // constant variable
            task.putvarbound(varIndex + 1, boundkey.fx, 0.5, 0.5);
            task.putvarname(varIndex + 1, "k" + i);

            // linear term variable
            task.putvarbound(varIndex + 2, boundkey.fr, -0.0, +0.0);
            task.putvarname(varIndex + 2, "r" + i);
            task.appendcons(1);
            for (int j = 0; j < n; j++) {
                task.putaij(conIndex, x[j], A[i][j]);
            }

            task.putaij(conIndex, varIndex + 2, -1);
            task.putaij(conIndex, v[i], -1);
            task.putconbound(conIndex, boundkey.fx, instance.y[i], instance.y[i]);
            conIndex++;

            task.appendcone(conetype.rquad, 0, new int[]{varIndex, varIndex + 1, varIndex + 2});
            varIndex += 3;

        }

    }

    /**
     * Builds an instance without a point. <br>
     *
     */
    void buildMinimizationStrong(double gamma, int k) {
//        task.dispose();

        int n = instance.n;
        int m = instance.m;
        double gammaDiv = gamma / (double) m;
        mLambda = 1000 * gammaDiv;
        varIndex = 0;
        conIndex = 0;
        x = new int[n + 1];
        z = new int[m];
        task.appendvars(n + 1 + m); // Continuous + auxiliary + discrete variables
//        task.appendcons(2 * n); // Big M constraints
        task.appendcons(1); // Cardinality
        int cardConstr = conIndex;
        task.putconbound(cardConstr, boundkey.up, 0, k);
        conIndex++;

        // Adds variables
        for (int i = 0; i < n; i++) {
            x[i] = varIndex;
            task.putvarbound(x[i], boundkey.fr, -0.0, +0.0);
            task.putvarname(x[i], "x" + i);
            varIndex++;
        }
        x[n] = varIndex;
        task.putvarbound(x[n], boundkey.fx, 1.0, 1.0);
        task.putvarname(x[n], "one");
        varIndex++;

        for (int i = 0; i < m; i++) {
            z[i] = varIndex;
            task.putvarbound(z[i], boundkey.ra, 0.0, 1.0);
            task.putvarname(z[i], "z" + i);
            task.putaij(cardConstr, z[i], 1);
            task.putvartype(z[i], variabletype.type_int);
            varIndex++;
        }

        double[][] A = instance.A;

        for (int i = 0; i < A.length; i++) {
            double[] row = new double[n + 1];
            System.arraycopy(A[i], 0, row, 0, n);
            row[n] = -instance.y[i];
            RealVector a = MatrixUtils.createRealVector(row);
            RealMatrix r1 = a.outerProduct(a);
            double norm2 = Math.pow(a.getNorm(), 2);
            double[][] diag = new double[n + 1][n + 1];
            for (int j = 0; j < n + 1; j++) {
                diag[j][j] = norm2 + gammaDiv;
            }
            double denom = gammaDiv * (gammaDiv + norm2);
            RealMatrix matrix = MatrixUtils.createRealMatrix(diag);
            matrix = matrix.subtract(r1);
            CholeskyDecomposition decomp = new CholeskyDecomposition(matrix);
            double[][] L = decomp.getL().getData();

            task.appendvars(4 + n);
            int tau1 = varIndex;
            task.putvarbound(tau1, boundkey.lo, 0, +0.0);
            task.putvarname(tau1, "tau1_" + i);
            task.putcj(tau1, 1);
            varIndex++;
            int tau2 = varIndex;
            task.putvarbound(tau2, boundkey.lo, 0, +0.0);
            task.putvarname(tau2, "tau2_" + i);
            task.putcj(tau2, 1);
            varIndex++;
            int s = varIndex;
            task.putvarbound(s, boundkey.fr, -0.0, +0.0);
            task.putvarname(s, "s_" + i);
            varIndex++;
            int[] w = new int[n + 1];
            for (int j = 0; j < n + 1; j++) {
                w[j] = varIndex;
                task.putvarbound(w[j], boundkey.fr, -0.0, +0.0);
                task.putvarname(w[j], "w_" + i + "," + j);
                varIndex++;
            }

            task.appendcons(n + 1);
            for (int j = 0; j < n + 1; j++) {
                task.putaij(conIndex, x[j], -1);
                task.putaij(conIndex, s, a.getEntry(j));
                for (int l = 0; l < n + 1; l++) {
                    task.putaij(conIndex, w[l], L[j][l]);
                }

                task.putconbound(conIndex, boundkey.fx, 0, 0);
                conIndex++;
            }

            // conic constraint 1
            task.appendvars(1);
            task.putvarbound(varIndex, boundkey.fx, 0.5 / denom, 0.5 / denom);
            task.putvarname(varIndex, "fix1" + i);
            int[] vars = new int[n + 3];
            vars[0] = varIndex;
            vars[1] = tau1;
            System.arraycopy(w, 0, vars, 2, n + 1);
            task.appendcone(conetype.rquad, 0, vars);
            varIndex++;

            // conic constraint 2
            task.appendvars(1);
            task.appendcons(1);
            task.putvarbound(varIndex, boundkey.lo, 0, +0.0);
            task.putvarname(varIndex, "fix2" + i);
            task.putarow(conIndex, new int[]{varIndex, z[i]}, new double[]{1, -0.5 / denom});
            task.putconbound(conIndex, boundkey.fx, 0, 0);

            task.appendcone(conetype.rquad, 0, new int[]{tau2, varIndex, s});
            varIndex++;
            conIndex++;
        }
        task.putcfix(-gamma);
    }

    /**
     *
     *
     */
    void buildMinimizationStronger(double gamma, int k) {
//        task.dispose();

        int n = instance.n;
        int m = instance.m;
        double gammaDiv = gamma / (double) m;
        mLambda = 1000 * gammaDiv;
        varIndex = 0;
        conIndex = 0;
        x = new int[n + 1];
        z = new int[m];
        task.appendvars(n + 1 + m); // Continuous + auxiliary + discrete variables
//        task.appendcons(2 * n); // Big M constraints
        task.appendcons(1); // Cardinality
        int cardConstr = conIndex;
        task.putconbound(cardConstr, boundkey.fx, k, k);
        conIndex++;

        // Adds variables
        for (int i = 0; i < n; i++) {
            x[i] = varIndex;
            task.putvarbound(x[i], boundkey.fr, -0.0, +0.0);
            task.putvarname(x[i], "x" + i);
            varIndex++;
        }
        x[n] = varIndex;
        task.putvarbound(x[n], boundkey.fx, 1.0, 1.0);
        task.putvarname(x[n], "one");
        varIndex++;

        for (int i = 0; i < m; i++) {
            z[i] = varIndex;
            task.putvarbound(z[i], boundkey.ra, 0.0, 1.0);
            task.putvarname(z[i], "z" + i);
            task.putaij(cardConstr, z[i], 1);
            if (isInt) {
                task.putvartype(z[i], variabletype.type_int);
            }

            varIndex++;
        }

        double[][] A = instance.A;

        for (int i = 0; i < A.length; i++) {
            double[] row = new double[n + 1];
            System.arraycopy(A[i], 0, row, 0, n);
            row[n] = -instance.y[i];
            RealVector a = MatrixUtils.createRealVector(row);
            double norm2 = 0;
            for (int j = 0; j < n; j++) {
                norm2 += row[j] * row[j] / gammaDiv;
            }
            norm2 += row[n] * row[n] / mLambda;
            double[][] diag = new double[n + 1][n + 1];
            for (int j = 0; j < n; j++) {
                diag[j][j] = 1 / gammaDiv;
            }
            diag[n][n] = 1 / mLambda;
            double denom = (1 + norm2);
            RealMatrix matrix = MatrixUtils.createRealMatrix(diag);
            RealVector a2 = matrix.operate(a);
            RealMatrix r1 = a2.outerProduct(a2);
            matrix = matrix.scalarMultiply(denom);
            matrix = matrix.subtract(r1);
            CholeskyDecomposition decomp = new CholeskyDecomposition(matrix);
            double[][] L = decomp.getL().getData();

            task.appendvars(4 + n);
            int tau1 = varIndex;
            task.putvarbound(tau1, boundkey.lo, 0, +0.0);
            task.putvarname(tau1, "tau1_" + i);
            task.putcj(tau1, 1);
            varIndex++;
            int tau2 = varIndex;
            task.putvarbound(tau2, boundkey.lo, 0, +0.0);
            task.putvarname(tau2, "tau2_" + i);
            task.putcj(tau2, 1);
            varIndex++;
            int s = varIndex;
            task.putvarbound(s, boundkey.fr, -0.0, +0.0);
            task.putvarname(s, "s_" + i);
            varIndex++;
            int[] w = new int[n + 1];
            for (int j = 0; j < n + 1; j++) {
                w[j] = varIndex;
                task.putvarbound(w[j], boundkey.fr, -0.0, +0.0);
                task.putvarname(w[j], "w_" + i + "," + j);
                varIndex++;
            }

            task.appendcons(n + 1);
            for (int j = 0; j < n + 1; j++) {
                task.putaij(conIndex, x[j], -1);
                task.putaij(conIndex, s, a2.getEntry(j));
                for (int l = 0; l < n + 1; l++) {
                    task.putaij(conIndex, w[l], L[j][l]);
                }

                task.putconbound(conIndex, boundkey.fx, 0, 0);
                conIndex++;
            }

            // conic constraint 1
            task.appendvars(1);
            task.putvarbound(varIndex, boundkey.fx, 0.5 / denom, 0.5 / denom);
            task.putvarname(varIndex, "fix1" + i);
            int[] vars = new int[n + 3];
            vars[0] = varIndex;
            vars[1] = tau1;
            System.arraycopy(w, 0, vars, 2, n + 1);
            task.appendcone(conetype.rquad, 0, vars);
            varIndex++;

            // conic constraint 2
            task.appendvars(1);
            task.appendcons(1);
            task.putvarbound(varIndex, boundkey.lo, 0, +0.0);
            task.putvarname(varIndex, "fix2" + i);
            task.putarow(conIndex, new int[]{varIndex, z[i]}, new double[]{1, -0.5 / denom});
            task.putconbound(conIndex, boundkey.fx, 0, 0);

            task.appendcone(conetype.rquad, 0, new int[]{tau2, varIndex, s});
            varIndex++;
            conIndex++;
        }
        task.putcfix(-mLambda * (double) m);
    }

    /**
     *
     *
     */
    void buildMinimizationFinal2(double[][] lambda, int k) {
//        task.dispose();

        int n = instance.n;
        int m = instance.m;
        varIndex = 0;
        conIndex = 0;
        x = new int[n];
        z = new int[m];
        task.appendvars(n + m); // Continuous  + discrete variables
//        task.appendcons(2 * n); // Big M constraints
        task.appendcons(1); // Cardinality
        int cardConstr = conIndex;
        task.putconbound(cardConstr, boundkey.fx, k, k);
        conIndex++;

        // Adds variables
        for (int i = 0; i < n; i++) {
            x[i] = varIndex;
            task.putvarbound(x[i], boundkey.fr, -0.0, +0.0);
            task.putvarname(x[i], "x" + i);
            varIndex++;
        }

        for (int i = 0; i < m; i++) {
            z[i] = varIndex;
            task.putvarbound(z[i], boundkey.ra, 0.0, 1.0);
            task.putvarname(z[i], "z" + i);
            task.putaij(cardConstr, z[i], 1);
            if (isInt) {
                task.putvartype(z[i], variabletype.type_int);
            }

            varIndex++;
        }

        double[][] A = instance.A;
        double objConstant = 0;
        double[] xObjCoef = new double[n];
        int tau1, tau2;
        v = new int[A.length];
        for (int i = 0; i < A.length; i++) {
            objConstant += instance.y[i] * instance.y[i];
            for (int j = 0; j < n; j++) {
                xObjCoef[j] -= 2 * instance.y[i] * A[i][j];
            }
            task.appendvars(3);
            tau1 = varIndex;
            task.putvarbound(tau1, boundkey.lo, 0.0, +0.0);
            task.putvarname(tau1, "tau1_" + i);
            task.putcj(tau1, 1);
            varIndex++;
            tau2 = varIndex;
            task.putvarbound(tau2, boundkey.lo, 0.0, +0.0);
            task.putvarname(tau2, "tau2_" + i);
            task.putcj(tau2, 1);
            varIndex++;
            v[i] = varIndex;
            task.putvarbound(v[i], boundkey.fr, 0.0, +0.0);
            task.putvarname(v[i], "v" + i);
            task.putcj(v[i], 2 * instance.y[i]);
            varIndex++;

            double[][] U = new double[n][n], W = new double[n + 1][n + 1];
            for (int j = 0; j < n; j++) {
                U[j][j] = lambda[i][j] + A[i][j] * A[i][j];
                W[j][j] = U[j][j];
                for (int l = j + 1; l < n; l++) {
                    U[j][l] = A[i][j] * A[i][l];
                    U[l][j] = U[j][l];
                    W[j][l] = U[j][l];
                    W[l][j] = W[j][l];
                }
                W[j][n] = -A[i][j];
                W[n][j] = W[j][n];
            }
            W[n][n] = 1;
            RealMatrix matrixU = MatrixUtils.createRealMatrix(U);
            RealMatrix matrixW = MatrixUtils.createRealMatrix(W);

            matrixU = MatrixUtils.inverse(matrixU);
            matrixW = MatrixUtils.inverse(matrixW);
            //Symmetrize
            for (int j = 0; j < U.length; j++) {
                for (int l = 0; l < U[j].length; l++) {
                    matrixU.setEntry(j, l, (matrixU.getEntry(j, l) + matrixU.getEntry(l, j)) / 2.0);
                    matrixU.setEntry(l, j, matrixU.getEntry(j, l));
                }
            }

            CholeskyDecomposition decompU = new CholeskyDecomposition(matrixU);
            for (int j = 0; j < W.length; j++) {
                for (int l = 0; l < W[j].length; l++) {
                    matrixW.setEntry(j, l, (matrixW.getEntry(j, l) + matrixW.getEntry(l, j)) / 2.0);
                    matrixW.setEntry(l, j, matrixW.getEntry(j, l));
                }
            }
//            System.out.println("");
            CholeskyDecomposition decompW = new CholeskyDecomposition(matrixW);
            double[][] CholU = decompU.getL().getData();
            double[][] CholW = decompW.getL().getData();

            task.appendvars(2 * n + 1);
            int[] u = new int[n];
            int[] w = new int[n + 1];
            for (int j = 0; j < n; j++) {
                u[j] = varIndex;
                task.putvarbound(u[j], boundkey.fr, -0.0, +0.0);
                task.putvarname(u[j], "u_" + i + "," + j);
                varIndex++;
                w[j] = varIndex;
                task.putvarbound(w[j], boundkey.fr, -0.0, +0.0);
                task.putvarname(w[j], "w_" + i + "," + j);
                varIndex++;
            }
            w[n] = varIndex;
            task.putvarbound(w[n], boundkey.fr, -0.0, +0.0);
            task.putvarname(w[n], "w_" + i + "," + n);
            varIndex++;

            task.appendcons(n + 1);
            for (int j = 0; j < n; j++) {
                task.putaij(conIndex, x[j], -1);
                for (int l = 0; l < n; l++) {
                    task.putaij(conIndex, u[l], CholU[j][l]);
                }
                for (int l = 0; l < n + 1; l++) {
                    task.putaij(conIndex, w[l], CholW[j][l]);
                }
                task.putconbound(conIndex, boundkey.fx, 0, 0);
                conIndex++;
            }

            task.putaij(conIndex, v[i], -1);
            for (int l = 0; l < n + 1; l++) {
                task.putaij(conIndex, w[l], CholW[n][l]);
            }
            task.putconbound(conIndex, boundkey.fx, 0, 0);
            conIndex++;

            // conic constraint 1
            task.appendvars(1);
            task.appendcons(1);
            task.putvarbound(varIndex, boundkey.lo, 0, 0);
            task.putvarname(varIndex, "(1-z)" + i);
            task.putarow(conIndex, new int[]{varIndex, z[i]}, new double[]{1, 0.5});
            task.putconbound(conIndex, boundkey.fx, 0.5, 0.5);

            int[] vars = new int[n + 2];
            vars[0] = varIndex;
            vars[1] = tau1;
            System.arraycopy(u, 0, vars, 2, n);
            task.appendcone(conetype.rquad, 0, vars);
            varIndex++;
            conIndex++;

            // conic constraint 2
            task.appendvars(1);
            task.appendcons(1);
            task.putvarbound(varIndex, boundkey.lo, 0, 0);
            task.putvarname(varIndex, "zAux" + i);
            task.putarow(conIndex, new int[]{varIndex, z[i]}, new double[]{1, -0.5});
            task.putconbound(conIndex, boundkey.fx, 0, 0);

            int[] vars2 = new int[n + 3];
            vars2[0] = varIndex;
            vars2[1] = tau2;
            System.arraycopy(w, 0, vars2, 2, w.length);
            task.appendcone(conetype.rquad, 0, vars2);
            varIndex++;
            conIndex++;

        }
        task.putcfix(objConstant);
        for (int j = 0; j < n; j++) {
            task.putcj(x[j], xObjCoef[j]);
        }
    }

    /**
     *
     *
     */
    void buildMinimizationFinal3(double[][] lambda, int k) {
//        task.dispose();

        int n = instance.n;
        int m = instance.m;
        varIndex = 0;
        conIndex = 0;
        x = new int[n];
        z = new int[m];
        task.appendvars(n + m); // Continuous  + discrete variables
//        task.appendcons(2 * n); // Big M constraints
        task.appendcons(1); // Cardinality
        int cardConstr = conIndex;
        task.putconbound(cardConstr, boundkey.fx, k, k);
        conIndex++;

        // Adds variables
        for (int i = 0; i < n; i++) {
            x[i] = varIndex;
            task.putvarbound(x[i], boundkey.fr, -0.0, +0.0);
            task.putvarname(x[i], "x" + i);
            varIndex++;
        }

        for (int i = 0; i < m; i++) {
            z[i] = varIndex;
            task.putvarbound(z[i], boundkey.ra, 0.0, 1.0);
            task.putvarname(z[i], "z" + i);
            task.putaij(cardConstr, z[i], 1);
            if (isInt) {
                task.putvartype(z[i], variabletype.type_int);
            }

            varIndex++;
        }

        double[][] A = instance.A;
        double objConstant = 0;
        double[] xObjCoef = new double[n];
        int tau1, tau2;
        v = new int[A.length];
        for (int i = 0; i < A.length; i++) {
            objConstant += instance.y[i] * instance.y[i];
            for (int j = 0; j < n; j++) {
                xObjCoef[j] -= 2 * instance.y[i] * A[i][j];
            }
            task.appendvars(3);
            tau1 = varIndex;
            task.putvarbound(tau1, boundkey.lo, 0.0, +0.0);
            task.putvarname(tau1, "tau1_" + i);
            task.putcj(tau1, 1);
            varIndex++;
            tau2 = varIndex;
            task.putvarbound(tau2, boundkey.lo, 0.0, +0.0);
            task.putvarname(tau2, "tau2_" + i);
            task.putcj(tau2, 1);
            varIndex++;
            v[i] = varIndex;
            task.putvarbound(v[i], boundkey.fr, 0.0, +0.0);
            task.putvarname(v[i], "v" + i);
            task.putcj(v[i], 2 * instance.y[i]);
            varIndex++;

            double denom = 1;
            double[] DInvA = new double[n];
            for (int j = 0; j < n; j++) {
                denom += instance.A[i][j] * instance.A[i][j] / lambda[i][j];
                DInvA[j] = instance.A[i][j] / lambda[i][j];
            }

            double[][] U = new double[n][n];
            for (int j = 0; j < n; j++) {
                U[j][j] = 1 / lambda[i][j] - DInvA[j] * DInvA[j] / denom;
                for (int l = j + 1; l < n; l++) {
                    U[j][l] = -DInvA[j] * DInvA[l] / denom;
                    U[l][j] = U[j][l];

                }
            }
            RealMatrix matrixU = MatrixUtils.createRealMatrix(U);

            CholeskyDecomposition decompU = new CholeskyDecomposition(matrixU);

//            System.out.println("");
            double[][] CholU = decompU.getL().getData();

            task.appendvars(n);
            int[] u = new int[n];
            for (int j = 0; j < n; j++) {
                u[j] = varIndex;
                task.putvarbound(u[j], boundkey.fr, -0.0, +0.0);
                task.putvarname(u[j], "u_" + i + "," + j);
                varIndex++;
            }

            task.appendcons(n);
            for (int j = 0; j < n; j++) {
                task.putaij(conIndex, x[j], -1);
                for (int l = 0; l < n; l++) {
                    task.putaij(conIndex, u[l], CholU[j][l]);
                }
                task.putaij(conIndex, v[i], DInvA[j] / denom);
                task.putconbound(conIndex, boundkey.fx, 0, 0);
                conIndex++;
            }

            // conic constraint 1
            task.appendvars(1);
            task.putvarbound(varIndex, boundkey.fx, 0.5, 0.5);
            task.putvarname(varIndex, "half" + i);

            int[] vars = new int[n + 2];
            vars[0] = varIndex;
            vars[1] = tau1;
            System.arraycopy(u, 0, vars, 2, n);
            task.appendcone(conetype.rquad, 0, vars);
            varIndex++;

            // conic constraint 2
            task.appendvars(1);
            task.appendcons(1);
            task.putvarbound(varIndex, boundkey.lo, 0, 0);
            task.putvarname(varIndex, "zAux" + i);
            task.putarow(conIndex, new int[]{varIndex, z[i]}, new double[]{1, -0.5 * denom});
            task.putconbound(conIndex, boundkey.fx, 0, 0);

            int[] vars2 = new int[3];
            vars2[0] = varIndex;
            vars2[1] = tau2;
            vars2[2] = v[i];
            task.appendcone(conetype.rquad, 0, vars2);
            varIndex++;
            conIndex++;

        }
        task.putcfix(objConstant);
        for (int j = 0; j < n; j++) {
            task.putcj(x[j], xObjCoef[j]);
        }
    }

    /**
     *
     *
     */
    void buildMinimizationFinal(double[][] lambda, int k) {
//        task.dispose();

        int n = instance.n;
        int m = instance.m;
        varIndex = 0;
        conIndex = 0;
        x = new int[n];
        z = new int[m];
        task.appendvars(n + m); // Continuous  + discrete variables
//        task.appendcons(2 * n); // Big M constraints
        task.appendcons(1); // Cardinality
        int cardConstr = conIndex;
        task.putconbound(cardConstr, boundkey.fx, k, k);
        conIndex++;

        // Adds variables
        for (int i = 0; i < n; i++) {
            x[i] = varIndex;
            task.putvarbound(x[i], boundkey.fr, -0.0, +0.0);
            task.putvarname(x[i], "x" + i);
            varIndex++;
        }

        for (int i = 0; i < m; i++) {
            z[i] = varIndex;
            task.putvarbound(z[i], boundkey.ra, 0.0, 1.0);
            task.putvarname(z[i], "z" + i);
            task.putaij(cardConstr, z[i], 1);
            if (isInt) {
                task.putvartype(z[i], variabletype.type_int);
            }

            varIndex++;
        }

        double[][] A = instance.A;
        double[][] AtA = instance.getAtA();
        double[][] bigQ = new double[n + m][n + m];
        for (int i = 0; i < n; i++) {
            bigQ[i][i] = AtA[i][i];
            for (int j = 0; j < m; j++) {
                bigQ[i][i] += lambda[j][i];
            }
            for (int j = i + 1; j < n; j++) {
                bigQ[i][j] = AtA[i][j];
                bigQ[j][i] = bigQ[i][j];
            }
            for (int j = 0; j < m; j++) {
                bigQ[i][n + j] = -A[j][i];
                bigQ[n + j][i] = bigQ[i][n + j];
            }
        }

        double objConstant = 0;
        double[] xObjCoef = new double[n];
        tau = new int[A.length];
        v = new int[A.length];
        for (int i = 0; i < A.length; i++) {
            objConstant += instance.y[i] * instance.y[i];
            for (int j = 0; j < n; j++) {
                xObjCoef[j] -= 2 * instance.y[i] * A[i][j];
            }

            double denom = 1;
            double[] DInvA = new double[n];
            for (int j = 0; j < n; j++) {
                denom += instance.A[i][j] * instance.A[i][j] / lambda[i][j];
                DInvA[j] = instance.A[i][j] / lambda[i][j];
            }

            task.appendvars(2);
            v[i] = varIndex;
            task.putvarbound(v[i], boundkey.fr, 0.0, +0.0);
            task.putvarname(v[i], "v" + i);
            task.putcj(v[i], 2 * instance.y[i]);
            varIndex++;
            tau[i] = varIndex;
            task.putvarbound(tau[i], boundkey.lo, 0.0, +0.0);
            task.putvarname(tau[i], "tau" + i);
            task.putcj(tau[i], (1.0 - EPSILON) / denom);
            varIndex++;

            double[][] U = new double[n][n];
            for (int j = 0; j < n; j++) {
                U[j][j] = 1 / lambda[i][j] - DInvA[j] * DInvA[j] / denom;
                for (int l = j + 1; l < n; l++) {
                    U[j][l] = -DInvA[j] * DInvA[l] / denom;
                    U[l][j] = U[j][l];

                }
            }
            RealMatrix matrixU = MatrixUtils.createRealMatrix(U);

            CholeskyDecomposition decompU = new CholeskyDecomposition(matrixU);

//            System.out.println("");
            double[][] LInv = MatrixUtils.inverse(decompU.getL()).getData();
            double sum;
            for (int j = 0; j < LInv.length; j++) {
                sum = 0;
                for (int l = 0; l < LInv[j].length; l++) {
                    sum += LInv[j][l] * DInvA[l];
                }
                bigQ[n + i][n + i] += sum * sum;
            }
            bigQ[n + i][n + i] /= (denom * denom);
            bigQ[n + i][n + i] += EPSILON / denom;

            // conic constraint 2
            task.appendvars(1);
            task.appendcons(1);
            task.putvarbound(varIndex, boundkey.lo, 0, 0);
            task.putvarname(varIndex, "zAux" + i);
            task.putarow(conIndex, new int[]{varIndex, z[i]}, new double[]{1, -0.5});
            task.putconbound(conIndex, boundkey.fx, 0, 0);

            int[] vars = new int[3];
            vars[0] = varIndex;
            vars[1] = tau[i];
            vars[2] = v[i];
            task.appendcone(conetype.rquad, 0, vars);
            varIndex++;
            conIndex++;

        }
        RealMatrix matrixQ = MatrixUtils.createRealMatrix(bigQ);
//        for (int i = 0; i < n+m; i++) {
//            for (int j = 0; j < n+m; j++) {
//                System.out.print(matrixQ.getEntry(i, j)+ " ");
//            }
//            System.out.println("");
//        }

        CholeskyDecomposition decompQ = new CholeskyDecomposition(matrixQ);

        double[][] CholQ = decompQ.getL().getData();
        int[] coneQuad = new int[n + m + 2];
        task.appendvars(n + m + 2);

        coneQuad[0] = varIndex;
        coneQuad[1] = varIndex + 1;
        task.putvarbound(varIndex, boundkey.fx, 0.5, 0.5);
        task.putvarbound(varIndex + 1, boundkey.lo, 0, +0.0);
        task.putcj(varIndex + 1, 1);
        varIndex += 2;

        for (int i = 0; i < n + m; i++) {
            task.appendcons(1);
            coneQuad[2 + i] = varIndex;
            task.putvarbound(coneQuad[2 + i], boundkey.fr, 0, 0);
            task.putvarname(coneQuad[2 + i], "quad" + i);
            task.putaij(conIndex, coneQuad[2 + i], -1);

            for (int j = 0; j < n; j++) {
                task.putaij(conIndex, x[j], CholQ[j][i]);
            }
            for (int j = 0; j < m; j++) {
                task.putaij(conIndex, v[j], CholQ[j + n][i]);
            }
            task.putconbound(conIndex, boundkey.fx, 0, 0);
            varIndex++;
            conIndex++;
        }
        task.appendcone(conetype.rquad, 0, coneQuad);

        task.putcfix(objConstant);
        for (int j = 0; j < n; j++) {
            task.putcj(x[j], xObjCoef[j]);
        }
    }

    public double solve(double[] xSol, double[] zSol, double[] tauSol, double[] obj) {
//        task.writedata("./data.cbf");
        long time = System.currentTimeMillis();
        task.optimize();
        obj[2] = (System.currentTimeMillis() - time) / 1000;
        time += task.getdouinf(mosek.dinfitem.optimizer_time);

        mosek.solsta[] solsta = new mosek.solsta[1];

        task.getsolsta(mosek.soltype.itr, solsta);
//        System.out.println("Status: " + solsta[0]);
        switch (solsta[0]) {
            case optimal:
            //case near_optimal:
            case integer_optimal:
                //case near_integer_optimal:
                double[] solOpt = new double[varIndex];
//                    double[] R = new double[n*(n+1)/2];

                task.getxxslice(mosek.soltype.itr, 0, varIndex, solOpt);
//                    task.getbarxj(mosek.soltype.itr, /* Request the interior solution. */
//                            0,
//                            R);
                for (int i = 0; i < xSol.length; i++) {
                    xSol[i] = solOpt[x[i]];
                }
                for (int i = 0; i < zSol.length; i++) {
                    zSol[i] = solOpt[z[i]];
                }
                if (tauSol != null) {
                    for (int i = 0; i < tauSol.length; i++) {
                        tauSol[i] = solOpt[tau[i]];
                    }
                }

                obj[1] = task.getprimalobj(mosek.soltype.itr);

                return task.getprimalobj(mosek.soltype.itr);
            case dual_infeas_cer:
            case prim_infeas_cer:
                //case near_dual_infeas_cer:
                //case near_prim_infeas_cer:
                System.out.println("Primal or dual infeasibility certificate found.");
                return -1;
            case unknown:
                System.out.println("The status of the solution could not be determined.");
                return -1;
            default:
//                System.out.println(k+"\t"+point);
                System.out.println("Other solution status.");
                return -1;
        }
    }

    public double solveIP(double[] xSol, double[] zSol, double[] tauSol, double[] obj) {
        long time = System.currentTimeMillis();
        task.optimize();
        obj[2] = (System.currentTimeMillis() - time) / 1000;
        obj[3] = task.getintinf(mosek.iinfitem.mio_num_branch);
        time += task.getdouinf(mosek.dinfitem.optimizer_time);

        mosek.solsta[] solsta = new mosek.solsta[1];

        task.getsolsta(mosek.soltype.itg, solsta);
//        System.out.println("Status: " + solsta[0]);
        switch (solsta[0]) {
            case optimal:
            //case near_optimal:
            case integer_optimal:
            case prim_feas:
                //case near_integer_optimal:
                double[] solOpt = new double[varIndex];
//                    double[] R = new double[n*(n+1)/2];

                task.getxxslice(mosek.soltype.itg, 0, varIndex, solOpt);
//                    task.getbarxj(mosek.soltype.itr, /* Request the interior solution. */
//                            0,
//                            R);
                for (int i = 0; i < xSol.length; i++) {
                    xSol[i] = solOpt[x[i]];
                }
                for (int i = 0; i < zSol.length; i++) {
                    zSol[i] = solOpt[z[i]];
                }
                if (tauSol != null) {
                    for (int i = 0; i < tauSol.length; i++) {
                        tauSol[i] = solOpt[tau[i]];
                    }
                }
                obj[1] = task.getdouinf(mosek.dinfitem.mio_obj_bound);
                obj[0] = task.getdouinf(mosek.dinfitem.mio_obj_int);
//                System.out.println(task.getdouinf(mosek.dinfitem.mio_obj_bound)+"\t"+task.getdouinf(mosek.dinfitem.mio_obj_int));
//                System.out.println("Solution obj= "+task.getprimalobj(mosek.soltype.itg)+"\t"+bound[0]);
                return task.getprimalobj(mosek.soltype.itg);
            case dual_infeas_cer:
            case prim_infeas_cer:
                //case near_dual_infeas_cer:
                //case near_prim_infeas_cer:
                System.out.println("Primal or dual infeasibility certificate found.");
                return -1;
            case unknown:
                System.out.println("The status of the solution could not be determined.");
                return -1;
            default:
                System.out.println("Other solution status.");
                return -1;
        }
    }

}

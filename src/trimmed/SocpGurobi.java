/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package trimmed;

import com.gurobi.gurobi.GRB;
import com.gurobi.gurobi.GRBEnv;
import com.gurobi.gurobi.GRBException;
import com.gurobi.gurobi.GRBLinExpr;
import com.gurobi.gurobi.GRBModel;
import com.gurobi.gurobi.GRBQuadExpr;
import com.gurobi.gurobi.GRBVar;
import data.Regression;
import java.util.Arrays;
import org.apache.commons.math3.linear.CholeskyDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * Solves primal LTS problems using Gurobi. <br>
 * @author Andres Gomez
 */
public class SocpGurobi {

    static final double M = 1e3; // Value of the big-M constant

    GRBEnv env;
    GRBModel gurobi;
    Regression instance;
    GRBVar[] x, z, v, tau; // Array with the primal variables
    double lambda0;
    boolean isInt = true; // Whether to solve a MIO
    boolean hasIntercept; // Whether to use an intercept.

    /**
     * Constructor for "direct" formulations. <br>
     * @param instance Dataset to be used. <br>
     * @param k Number of outliers. <br>
     * @param gamma L2 regularization parameter. <br>
     * @param method Method to be used.
     * 1= big M.
     * 2= conic (but conic+ is in different constructor). 
     * 3= Least Median of Squares from Bertsimas and Mazumder.
     * 4= Least absolute deviations.
     * 5= Ridge regression. 
     * 6= Huber loss. <br>
     * @param isInt Whether to solve a MIO. <br>
     * @param hasIntercept Whether to use an intercept. <br>
     * @throws GRBException 
     */
    public SocpGurobi(Regression instance, int k, double gamma, int method, boolean isInt, boolean hasIntercept) throws GRBException {
//        gamma=gamma/2.0;
        this.instance = instance;
        this.isInt = isInt;
        this.hasIntercept = hasIntercept;

        lambda0 = gamma;
        switch (Math.abs(method)) {
            case 1:
                createCoreModel(k);
                buildMinimizationLTSBigM(gamma);
                break;
            case 2:
                createCoreModel(k);
                double[][] lambdaInit = new double[instance.m][instance.n];
                for (int i = 0; i < instance.m; i++) {
                    for (int j = 0; j < instance.n; j++) {
                        lambdaInit[i][j] = gamma / (double) instance.m;
                    }
//                lambdaInit[i][instance.n] = M * gamma / (double) instance.m;
                }
                buildMinimizationFinal(lambdaInit);
                break;
            case 3:
                createCoreModel(instance.m - k);
                buildMinimizationMedian();
                break;
            case 4:
                createCoreModel(k);
                buildLAD();
                break;
            case 5:
                createCoreModel(k);
                buildMinimizationOLS(lambda0);
                break;
            case 6:
                createCoreModel(k);
                buildHuber(lambda0);
                break;
            default:
                break;
        }

    }

    /**
     * Constructor for conic+ formulation.<br>
     * @param instance Dataset to be used. <br>
     * @param k Number of outliers. <br>
     * @param regularization Regularization parameters to be used. <br>
     * @param diagonal
     * @param objConstant
     * @param objTerms

     * @param isInt Whether to solve a MIO. <br>
     * @param hasIntercept Whether to use an intercept. <br>
     * @throws GRBException 
     */
    public SocpGurobi(Regression instance, int k, double[][] regularization, double[] diagonal, double objConstant, double[] objTerms, boolean isInt) throws GRBException {
//        gamma=gamma/2.0;
        this.instance = instance;
        this.isInt = isInt;

        createCoreModel(k);
        buildMinimizationDecomp(regularization, diagonal, objConstant, objTerms);

    }


    /**
     * Constructor for an instances where the binary variable controlling
     * outliers are fixed to predetermined values. <br>
     *
     * @param instance Regression instances. <br>
     * @param k Number of outliers. <br>
     * @param gamma L2 regularization. <br>
     * @param zVal Array representing a current solution. Indexes associated
     * with the k highest values will be fixed to 1, and the remaining will be
     * fixed to 0. <br>
     * @param hasIntercept Whether the problem has an intercept or not. <br>
     * @throws GRBException
     */
    public SocpGurobi(Regression instance, int k, double gamma, double[] zVal, boolean hasIntercept) throws GRBException {
        this.hasIntercept = hasIntercept;
        this.instance = instance;

        lambda0 = gamma;
        createCoreModel(k);
        buildMinimizationFixed(gamma, k, zVal);

    }

    /**
     * Builds a least absolute deviations formulation as an LP. <br>
     *
     * @throws GRBException
     */
    private void buildLAD() throws GRBException {
        int m = instance.m;

        GRBLinExpr obj = new GRBLinExpr();
        for (int i = 0; i < m; i++) {
            GRBLinExpr linear = new GRBLinExpr();
            linear.addConstant(-instance.y[i]);
            linear.addTerms(instance.A[i], x);
            GRBVar error = gurobi.addVar(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY,
                    0, GRB.CONTINUOUS, "e" + i);
            gurobi.addConstr(linear, GRB.LESS_EQUAL, error, "errG" + i);
            GRBLinExpr minusErr = new GRBLinExpr();
            minusErr.addTerm(-1, error);
            gurobi.addConstr(linear, GRB.GREATER_EQUAL, minusErr, "errL" + i);
            obj.addTerm(1, error);
        }

        gurobi.setObjective(obj, GRB.MINIMIZE);
    }

    /**
     * Builds a QP formulation to optimize the Huber loss. <br>
     *
     * @throws GRBException
     */
    private void buildHuber(double param) throws GRBException {
        int m = instance.m;

        GRBQuadExpr obj = new GRBQuadExpr();
        for (int i = 0; i < m; i++) {
            GRBLinExpr linear = new GRBLinExpr();
            GRBVar w = gurobi.addVar(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, 0, GRB.CONTINUOUS, "w" + i);
            GRBVar epi = gurobi.addVar(0, Double.POSITIVE_INFINITY, 0, GRB.CONTINUOUS, "abs_w" + i);
            gurobi.addConstr(w, GRB.LESS_EQUAL, epi, "abs1W" + i);
            GRBLinExpr minusW = new GRBLinExpr();
            minusW.addTerm(-1, w);
            gurobi.addConstr(minusW, GRB.LESS_EQUAL, epi, "abs2W" + i);

            linear.addConstant(-instance.y[i]);
            linear.addTerms(instance.A[i], x);
            linear.addTerm(1, w);
            GRBVar error = gurobi.addVar(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY,
                    0, GRB.CONTINUOUS, "e" + i);
            gurobi.addConstr(linear, GRB.EQUAL, error, "errG" + i);

            obj.addTerm(1, error, error);
            obj.addTerm(param, epi);
        }

        gurobi.setObjective(obj, GRB.MINIMIZE);
    }

    /**
     * Builds the formulation proposed in the paper ``least quantile regression
     * via modern optimization" by Bertsimas and Mazumder. <br>
     *
     * @throws GRBException
     */
    private void buildMinimizationMedian() throws GRBException {
        int m = instance.m;
        GRBVar gamma = gurobi.addVar(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, 0, GRB.CONTINUOUS, "gamma");
        for (int i = 0; i < m; i++) {
            GRBVar rP = gurobi.addVar(0, Double.POSITIVE_INFINITY, 0, GRB.CONTINUOUS, "r+" + i);
            GRBVar rM = gurobi.addVar(0, Double.POSITIVE_INFINITY, 0, GRB.CONTINUOUS, "r+" + i);
            GRBVar u = gurobi.addVar(0, Double.POSITIVE_INFINITY, 0, GRB.CONTINUOUS, "u" + i);
            GRBVar baru = gurobi.addVar(0, Double.POSITIVE_INFINITY, 0, GRB.CONTINUOUS, "bar_u" + i);
            GRBLinExpr quantile = new GRBLinExpr();
            quantile.addTerm(1, rP);
            quantile.addTerm(1, rM);
            quantile.addTerm(-1, gamma);
            quantile.addTerm(-1, baru);
            quantile.addTerm(1, u);
            gurobi.addConstr(quantile, GRB.EQUAL, 0, "quantile" + i);
            GRBLinExpr error = new GRBLinExpr();
            error.addTerm(1, rP);
            error.addTerm(-1, rM);
            error.addTerms(instance.A[i], x);
            gurobi.addConstr(error, GRB.EQUAL, instance.y[i], "error" + i);
            gurobi.addConstr(u, GRB.LESS_EQUAL, gamma, "obj" + i);
            gurobi.addSOS(new GRBVar[]{u, baru}, new double[]{1, 2}, GRB.SOS_TYPE1);
            gurobi.addSOS(new GRBVar[]{rP, rM}, new double[]{1, 2}, GRB.SOS_TYPE1);
            gurobi.addSOS(new GRBVar[]{z[i], u}, new double[]{1, 2}, GRB.SOS_TYPE1);
        }

        GRBLinExpr obj = new GRBLinExpr();
        obj.addTerm(1, gamma);
        gurobi.setObjective(obj, GRB.MINIMIZE);
    }

    /**
     * Builds a bigM formulation for LTS. <br>
     *
     * @param lambda L2 regularization parameter. <br>
     * @throws GRBException
     */
    private void buildMinimizationLTSBigM(double lambda) throws GRBException {
        int n = instance.n;
        int m = instance.m;

        // Creates auxiliary variables
        double[] lbW = new double[m], ubW = new double[m];
        char[] typeW = new char[m];
        String[] namesW = new String[m];
        for (int i = 0; i < m; i++) {
            lbW[i] = Double.NEGATIVE_INFINITY;
            ubW[i] = Double.POSITIVE_INFINITY;
            typeW[i] = GRB.CONTINUOUS;
            namesW[i] = "w" + i;

        }
        GRBVar[] w = gurobi.addVars(lbW, ubW, null, typeW, namesW);
        for (int i = 0; i < m; i++) {
//            GRBLinExpr exprW = new GRBLinExpr();
//            exprW.addTerm(1, w[i]);
//            gurobi.addGenConstrIndicator(z[i], 0, exprW, GRB.EQUAL, 0, "indicator" + i);

//           GRBVar ziCompl=gurobi.addVar(0, 1, 0, GRB.CONTINUOUS, "z"+i+"compl");
//           GRBLinExpr sum= new GRBLinExpr();
//           sum.addTerm(1, ziCompl);
//           sum.addTerm(1, z[i]);
//           gurobi.addConstr(sum, GRB.EQUAL, 1, "compl"+i);
//           gurobi.addSOS(new GRBVar[]{w[i], ziCompl}, new double[]{1, 2}, GRB.SOS_TYPE1);
            GRBLinExpr up = new GRBLinExpr(), down = new GRBLinExpr();
            up.addTerm(M, z[i]);
            gurobi.addConstr(w[i], GRB.LESS_EQUAL, up, "bigMU" + i);
            down.addTerm(-M, z[i]);
            gurobi.addConstr(w[i], GRB.GREATER_EQUAL, down, "bigMD" + i);
        }

        // Creates objective
        GRBQuadExpr obj = new GRBQuadExpr();
        for (int i = 0; i < n - 1; i++) {
            obj.addTerm(lambda, x[i], x[i]);
        }
        if (!hasIntercept) {
            obj.addTerm(lambda, x[n - 1], x[n - 1]);
        }
        for (int i = 0; i < m; i++) {
            GRBLinExpr linear = new GRBLinExpr();
            linear.addConstant(-instance.y[i]);
            linear.addTerms(instance.A[i], x);
            linear.addTerm(-1, w[i]);
            GRBVar error = gurobi.addVar(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY,
                    0, GRB.CONTINUOUS, "e" + i);
            gurobi.addConstr(linear, GRB.EQUAL, error, "err" + i);
            obj.addTerm(1, error, error);
        }

        gurobi.setObjective(obj, GRB.MINIMIZE);
    }

    /**
     * Builds a formulation for ridge regression, without outlier removal. <br>
     *
     * @param lambda L2 regularization parameter. <br>
     * @throws GRBException
     */
    private void buildMinimizationOLS(double lambda) throws GRBException {
        int n = instance.n;
        int m = instance.m;

        // Creates objective
        GRBQuadExpr obj = new GRBQuadExpr();
        for (int i = 0; i < n - 1; i++) {
            obj.addTerm(lambda, x[i], x[i]);
        }
        if (!hasIntercept) {
            obj.addTerm(lambda, x[n - 1], x[n - 1]);
        }

        for (int i = 0; i < m; i++) {
//            System.out.println(items[i].index+"\t"+items[i].val);

            GRBLinExpr linear = new GRBLinExpr();

            linear.addConstant(-instance.y[i]);
            linear.addTerms(instance.A[i], x);
            GRBVar error = gurobi.addVar(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY,
                    0, GRB.CONTINUOUS, "e" + i);
            gurobi.addConstr(linear, GRB.EQUAL, error, "err" + i);
            obj.addTerm(1, error, error);
        }

        gurobi.setObjective(obj, GRB.MINIMIZE);
    }

    /**
     * Builds a minimization problem with the discrete variables fixed.
     *
     * @param lambda L2 regularization. <br>
     * @param k Number of variables fixed to 1. <br>
     * @param zVal Array representing a current solution. Indexes associated
     * with the k highest values will be fixed to 1, and the remaining will be
     * fixed to 0. <br>
     * @throws GRBException
     */
    private void buildMinimizationFixed(double lambda, int k, double[] zVal) throws GRBException {
        int n = instance.n;
        int m = instance.m;
        Item[] items = new Item[zVal.length];
        for (int i = 0; i < zVal.length; i++) {
            items[i] = new Item(i, zVal[i]);
        }
        Arrays.sort(items);

        GRBQuadExpr obj = new GRBQuadExpr();
        for (int i = 0; i < n - 1; i++) {
            obj.addTerm(lambda, x[i], x[i]);
        }
        if (!hasIntercept) {
            obj.addTerm(lambda, x[n - 1], x[n - 1]);
        }
        for (int i = 0; i < k; i++) {
            gurobi.addConstr(z[items[i].index], GRB.EQUAL, 1, null);
//            System.out.println(items[i].index+"\t"+items[i].val);
        }
        for (int i = k; i < m; i++) {
//            System.out.println(items[i].index+"\t"+items[i].val);
            gurobi.addConstr(z[items[i].index], GRB.EQUAL, 0, null);
            GRBLinExpr linear = new GRBLinExpr();

            linear.addConstant(-instance.y[items[i].index]);
            linear.addTerms(instance.A[items[i].index], x);
            GRBVar error = gurobi.addVar(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY,
                    0, GRB.CONTINUOUS, "e" + i);
            gurobi.addConstr(linear, GRB.EQUAL, error, "err" + i);
            obj.addTerm(1, error, error);
        }

        gurobi.setObjective(obj, GRB.MINIMIZE);
    }

    /**
     * Conic formulation. <br>
     * @param lambda Regularization parameters.
     */
    private void buildMinimizationFinal(double[][] lambda) throws GRBException {
//        task.dispose();

        int n = instance.n;
        int m = instance.m;
        double[][] A = instance.A;
        double[][] AtA = instance.getAtA();
        double[][] bigQ = new double[n + m][n + m];
        GRBQuadExpr obj = new GRBQuadExpr();

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

        double[] lbV = new double[m], ubV = new double[m],
                lbTau = new double[m], ubTau = new double[m];
        char[] typeV = new char[m], typeTau = new char[m];
        String[] namesV = new String[m], namesTau = new String[m];
        for (int i = 0; i < m; i++) {
            lbV[i] = Double.NEGATIVE_INFINITY;
            ubV[i] = Double.POSITIVE_INFINITY;
            typeV[i] = GRB.CONTINUOUS;
            namesV[i] = "v" + i;
            lbTau[i] = 0;
            ubTau[i] = Double.POSITIVE_INFINITY;
            typeTau[i] = GRB.CONTINUOUS;
            namesTau[i] = "tau" + i;

        }
        v = gurobi.addVars(lbV, ubV, null, typeV, namesV);
        tau = gurobi.addVars(lbTau, ubTau, null, typeTau, namesTau);

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

            obj.addTerm(2 * instance.y[i], v[i]);
            obj.addTerm(1 / denom, tau[i]);

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

            GRBQuadExpr lhs = new GRBQuadExpr(), rhs = new GRBQuadExpr();
            lhs.addTerm(1, v[i], v[i]);
            rhs.addTerm(1, z[i], tau[i]);
            gurobi.addQConstr(lhs, GRB.LESS_EQUAL, rhs, "ConeW" + i);

        }

        for (int i = 0; i < n; i++) {
            obj.addTerm(bigQ[i][i], x[i], x[i]);
            for (int j = i + 1; j < n; j++) {
                obj.addTerm(2 * bigQ[i][j], x[i], x[j]);
            }
            for (int j = 0; j < m; j++) {
                obj.addTerm(2 * bigQ[i][n + j], x[i], v[j]);
            }
        }
        for (int i = 0; i < m; i++) {
            obj.addTerm(bigQ[n + i][n + i], v[i], v[i]);
        }

        obj.addConstant(objConstant);
        for (int j = 0; j < n; j++) {
            obj.addTerm(xObjCoef[j], x[j]);

        }
        gurobi.setObjective(obj, GRB.MINIMIZE);
    }

    /**
     * Builds formulation used for conic+ method.
     *
     */
    private void buildMinimizationDecomp(double[][] regularization, double[] diagonal, double constant, double[] terms) throws GRBException {
//        task.dispose();

        int n = instance.n;
        int m = instance.m;
        double[][] A = instance.A;
        double[][] AtA = instance.getAtA();
        double[][] bigQ = new double[n + m][n + m];
        double min = Double.POSITIVE_INFINITY;
        double max = 0;
        for (double d : diagonal) {
            min = Math.min(min, d);
            max = Math.max(max, d);
        }
        System.out.println("MIN/MAX " + min + "\t" + max);
        double epsilon = Math.min((1 - max) / 2.0, 1e-3);
        GRBQuadExpr obj = new GRBQuadExpr();

        for (int i = 0; i < n; i++) {
            bigQ[i][i] = AtA[i][i];
            bigQ[i][i] += regularization[i][i];

            for (int j = i + 1; j < n; j++) {
                bigQ[i][j] = AtA[i][j];
                bigQ[i][j] += regularization[i][j];
                bigQ[j][i] = bigQ[i][j];
            }
            for (int j = 0; j < m; j++) {
                bigQ[i][n + j] = -A[j][i];
                bigQ[n + j][i] = bigQ[i][n + j];
            }
        }

        for (int i = 0; i < m; i++) {
//            bigQ[i+n][i+n]=diagonal[i]+DecompMosekSDP.MIN_DIAG/2.0;
            bigQ[i + n][i + n] = diagonal[i] + epsilon;
//             bigQ[i+n][i+n]=1;
        }

        double objConstant = 0;
        double[] xObjCoef = new double[n];

        double[] lbV = new double[m], ubV = new double[m],
                lbTau = new double[m], ubTau = new double[m];
        char[] typeV = new char[m], typeTau = new char[m];
        String[] namesV = new String[m], namesTau = new String[m];
        for (int i = 0; i < m; i++) {
            lbV[i] = Double.NEGATIVE_INFINITY;
            ubV[i] = Double.POSITIVE_INFINITY;
            typeV[i] = GRB.CONTINUOUS;
            namesV[i] = "v" + i;
            lbTau[i] = 0;
            ubTau[i] = Double.POSITIVE_INFINITY;
            typeTau[i] = GRB.CONTINUOUS;
            namesTau[i] = "tau" + i;

        }
        v = gurobi.addVars(lbV, ubV, null, typeV, namesV);
        tau = gurobi.addVars(lbTau, ubTau, null, typeTau, namesTau);

        for (int i = 0; i < A.length; i++) {
            objConstant += instance.y[i] * instance.y[i];
            for (int j = 0; j < n; j++) {
                xObjCoef[j] -= 2 * instance.y[i] * A[i][j];
            }
            obj.addTerm(2 * instance.y[i], v[i]);

            GRBQuadExpr lhs = new GRBQuadExpr(), rhs = new GRBQuadExpr();
            lhs.addTerm(1, v[i], v[i]);
            rhs.addTerm(1, z[i], tau[i]);
            gurobi.addQConstr(lhs, GRB.LESS_EQUAL, rhs, "ConeW" + i);
            obj.addTerm(1 - diagonal[i] - epsilon, tau[i]);
//            obj.addTerm(1-diagonal[i], tau[i]);
        }

        for (int i = 0; i < n; i++) {
            obj.addTerm(bigQ[i][i], x[i], x[i]);
            for (int j = i + 1; j < n; j++) {
                obj.addTerm(2 * bigQ[i][j], x[i], x[j]);
            }
            for (int j = 0; j < m; j++) {
                obj.addTerm(2 * bigQ[i][n + j], x[i], v[j]);
            }
        }
        for (int i = 0; i < m; i++) {
            obj.addTerm(bigQ[n + i][n + i], v[i], v[i]);
        }

        obj.addConstant(objConstant + constant);
        for (int j = 0; j < n; j++) {
            obj.addTerm(xObjCoef[j] + terms[j], x[j]);

        }
        gurobi.setObjective(obj, GRB.MINIMIZE);
    }

    /**
     * Creates model and basic variables. <br>
     *
     * @param k Max number of outliers. <br>
     * @throws GRBException
     */
    private void createCoreModel(int k) throws GRBException {
        env = new GRBEnv();
        gurobi = new GRBModel(env);
        gurobi.set(GRB.DoubleParam.TimeLimit, 600);
//        gurobi.set(GRB.IntParam.Threads, 1);
//        gurobi.set(GRB.IntParam.LogToConsole,0);

        int n = instance.n;
        int m = instance.m;
        double[] lbX = new double[n], ubX = new double[n],
                lbZ = new double[m], ubZ = new double[m];
        char[] typeX = new char[n], typeZ = new char[m];
        String[] namesX = new String[n], namesZ = new String[m];
        for (int i = 0; i < n; i++) {
            lbX[i] = Double.NEGATIVE_INFINITY;
            ubX[i] = Double.POSITIVE_INFINITY;
            typeX[i] = GRB.CONTINUOUS;
            namesX[i] = "x" + i;
        }
        double[] ones = new double[m];
        for (int i = 0; i < m; i++) {
            lbZ[i] = 0;
            ubZ[i] = 1;
            if (isInt) {
                typeZ[i] = GRB.BINARY;
            } else {
                typeZ[i] = GRB.CONTINUOUS;
            }
            namesZ[i] = "z" + i;
            ones[i] = 1;
        }
        x = gurobi.addVars(lbX, ubX, null, typeX, namesX);
        z = gurobi.addVars(lbZ, ubZ, null, typeZ, namesZ);
        GRBLinExpr card = new GRBLinExpr();
        card.addTerms(ones, z);
        gurobi.addConstr(card, GRB.EQUAL, k, "Cardinality");
    }

    /**
     * Solves the problem and retrieves optimal values of original variables. <br>
     * @param xSol Array to store the optimal regression coefficients. <br>
     * @param zSol Array to store the optimal indicator variables controlling outliers. <br>
     * @param metrics Array to store the metrics of the solution: 
     * 0: best obj value found, 1: best lower bound, 2: time in seconds, 3: number of nodes. <br> 
     * @return The best objective value found
     * @throws GRBException 
     */
    double solve(double[] xSol, double[] zSol, double[] metrics) throws GRBException {
        long time = System.currentTimeMillis();
        gurobi.optimize();
        metrics[2] = (System.currentTimeMillis() - time) / 1000;
        metrics[3] = gurobi.get(GRB.DoubleAttr.NodeCount);

        int optimstatus = gurobi.get(GRB.IntAttr.Status);
        double objval;
        switch (optimstatus) {
            case GRB.Status.OPTIMAL: {
                double[] xVals = gurobi.get(GRB.DoubleAttr.X, x), zVals = gurobi.get(GRB.DoubleAttr.X, z);
                System.arraycopy(xVals, 0, xSol, 0, xSol.length);
                System.arraycopy(zVals, 0, zSol, 0, zSol.length);
                metrics[0] = gurobi.get(GRB.DoubleAttr.ObjVal);
                metrics[1] = gurobi.get(GRB.DoubleAttr.ObjBound);
                System.out.println("Solution=");
                System.out.println("\t " + gurobi.get(GRB.DoubleAttr.ObjVal) + "\t" + gurobi.get(GRB.DoubleAttr.ObjBound) + "\t" + gurobi.get(GRB.DoubleAttr.MIPGap));
                objval = gurobi.get(GRB.DoubleAttr.ObjVal);
                break;
            }
            case GRB.Status.INF_OR_UNBD:
                System.out.println("Model is infeasible or unbounded");
                return -1;
            case GRB.Status.INFEASIBLE:
                System.out.println("Model is infeasible");
                return -1;
            case GRB.Status.UNBOUNDED:
                System.out.println("Model is unbounded");
                return -1;
            default: {
                System.out.println("Optimization was stopped with status = "
                        + optimstatus);
                double[] xVals = gurobi.get(GRB.DoubleAttr.X, x), zVals = gurobi.get(GRB.DoubleAttr.X, z);
                System.arraycopy(xVals, 0, xSol, 0, xSol.length);
                System.arraycopy(zVals, 0, zSol, 0, zSol.length);
                metrics[0] = gurobi.get(GRB.DoubleAttr.ObjVal);
                metrics[1] = gurobi.get(GRB.DoubleAttr.ObjBound);
                System.out.println("Solution=");
                System.out.println("\t " + gurobi.get(GRB.DoubleAttr.ObjVal) + "\t" + gurobi.get(GRB.DoubleAttr.ObjBound) + "\t" + gurobi.get(GRB.DoubleAttr.MIPGap));
                objval = gurobi.get(GRB.DoubleAttr.ObjVal);
                break;
            }
        }

        return objval;
    }

    /**
     * Solves the problem and retrieves optimal values of original and auxiliary variables. <br>
     * @param xSol Array to store the optimal regression coefficients. <br>
     * @param zSol Array to store the optimal indicator variables controlling outliers. <br>
     * @param vSol Array to store the auxiliary v variables. <br>
     * @param tauSol Array to store the auxiliary tau variables. <br>
     * @param metrics Array to store the metrics of the solution: 
     * 0: best obj value found, 1: best lower bound, 2: time in seconds, 3: number of nodes. <br> 
     * @return The best objective value found
     * @throws GRBException 
     */
    double solve(double[] xSol, double[] zSol, double[] vSol, double[] tauSol, double[] metrics) throws GRBException {
        long time = System.currentTimeMillis();
        gurobi.write("./model.lp");
        gurobi.optimize();
        metrics[2] = (System.currentTimeMillis() - time) / 1000;
        metrics[3] = gurobi.get(GRB.DoubleAttr.NodeCount);

        int optimstatus = gurobi.get(GRB.IntAttr.Status);
        double objval;
        switch (optimstatus) {
            case GRB.Status.OPTIMAL: {
                double[] xVals = gurobi.get(GRB.DoubleAttr.X, x), zVals = gurobi.get(GRB.DoubleAttr.X, z),
                        vVals = gurobi.get(GRB.DoubleAttr.X, v), tauVals = gurobi.get(GRB.DoubleAttr.X, tau);
                System.arraycopy(xVals, 0, xSol, 0, xSol.length);
                System.arraycopy(zVals, 0, zSol, 0, zSol.length);
                System.arraycopy(vVals, 0, vSol, 0, vSol.length);
                System.arraycopy(tauVals, 0, tauSol, 0, tauSol.length);
                metrics[0] = gurobi.get(GRB.DoubleAttr.ObjVal);
                if (isInt) {
                    metrics[1] = gurobi.get(GRB.DoubleAttr.ObjBound);
                    System.out.println("Solution=");
                    System.out.println("\t " + gurobi.get(GRB.DoubleAttr.ObjVal) + "\t" + gurobi.get(GRB.DoubleAttr.ObjBound) + "\t" + gurobi.get(GRB.DoubleAttr.MIPGap));
                } else {
                    metrics[1] = gurobi.get(GRB.DoubleAttr.ObjVal);
                }       //            System.out.println("Solution=");
//            System.out.println("\t " + gurobi.get(GRB.DoubleAttr.ObjVal) + "\t" + gurobi.get(GRB.DoubleAttr.ObjBound) + "\t" + gurobi.get(GRB.DoubleAttr.MIPGap));
                objval = gurobi.get(GRB.DoubleAttr.ObjVal);
                break;
            }
            case GRB.Status.INF_OR_UNBD:
                System.out.println("Model is infeasible or unbounded");
                return -1;
            case GRB.Status.INFEASIBLE:
                System.out.println("Model is infeasible");
                return -1;
            case GRB.Status.UNBOUNDED:
                System.out.println("Model is unbounded");
                return -1;
            default: {
                System.out.println("Optimization was stopped with status = "
                        + optimstatus);
                double[] xVals = gurobi.get(GRB.DoubleAttr.X, x), zVals = gurobi.get(GRB.DoubleAttr.X, z);
                System.arraycopy(xVals, 0, xSol, 0, xSol.length);
                System.arraycopy(zVals, 0, zSol, 0, zSol.length);
                metrics[0] = gurobi.get(GRB.DoubleAttr.ObjVal);
                metrics[1] = gurobi.get(GRB.DoubleAttr.ObjBound);
                System.out.println("Solution=");
                System.out.println("\t " + gurobi.get(GRB.DoubleAttr.ObjVal) + "\t" + gurobi.get(GRB.DoubleAttr.ObjBound) + "\t" + gurobi.get(GRB.DoubleAttr.MIPGap));
                objval = gurobi.get(GRB.DoubleAttr.ObjVal);
                break;
            }
        }

        return objval;
    }

    /**
     * Auxiliary class to sort an array in descending order. 
     */
    private class Item implements Comparable<Item> {

        final int index;
        final double val;

        public Item(int index, double val) {
            this.index = index;
            this.val = val;
        }

        @Override
        public int compareTo(Item o) {
            return -Double.compare(val, o.val);
        }

    }

}

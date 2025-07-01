/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package trimmed;

import com.gurobi.gurobi.GRB;
import com.gurobi.gurobi.GRBException;
import data.Regression;
import java.util.ArrayList;
import java.util.List;

/**
 * Runs method to find an optimal decomposition based on a primal dual method. 
 * @author Andres Gomez.
 */
public class PrimalDualSDP {

    final double epsilon = 1e-6;
    double regularization, interceptBase; // Regularization parameter and target intercept
    int k; // Number of outliers
    Regression instance; // Object with the instance to be solved
    boolean mip,intercept; // Whether to solve a MIO,and whether to use an intercept

    /**
     * Constructor by parameters. <br>
     * @param regularization Regularization parameter. <br>
     * @param k Number of outliers. <br>
     * @param instance Instance to be solved. <br>
     * @param mip Whether to solve a MIO. <br>
     * @param intercept Whether to use an intercept. <br>
     * @param interceptBase If using an intercept, the target value.
     */
    public PrimalDualSDP(double regularization, int k, Regression instance,
            boolean mip, boolean intercept, double interceptBase) {
        this.regularization = regularization;
        this.k = k;
        this.instance = instance;
        this.mip = mip;
        this.intercept=intercept;
        this.interceptBase=interceptBase;

    }

    /**
     * Solves the problem via the primal-dual procedure + (potentially) a MIO
     * @param xSol Array to store the solution for the continuous variables. <br>
     * @param zSol Array to store the solution for the indicator variables. <br>
     * @param metrics Array to store the execution metrics.
     * 0= best objective value found.
     * 1= best bound.
     * 2= time.
     * 3= total number of branch and bound nodes.
     * @throws GRBException 
     */
    public void solve(double[] xSol, double[] zSol, double[] metrics) throws GRBException {
        int n = instance.n;
        int m = instance.m;

        int iter = 0, iterLimit = 20;
        double lb = 0, ub = Double.POSITIVE_INFINITY, ubFirst = ub;
        List<Double> gaps = new ArrayList<>();

        double[][] reg = new double[n][n];
        for (int i = 0; i < n; i++) {
            reg[i][i] = regularization;
        }
       

        double constant=0;
        double[] terms= new double[n];
        if(intercept)
        {
            constant=regularization*interceptBase*interceptBase;
            terms[n-1]=-2*regularization*interceptBase;
        }
        
        DecompMosekSDP dual = new DecompMosekSDP(instance, reg);
        double[] diag = new double[m];
        dual.solve(diag);
        System.out.println("Diagonal=");
        for (double d : diag) {
            System.out.print(d + " ");
        }
        System.out.println("");
        System.out.println("");
        System.out.println("------------------------------------------------------");
//        SocpMosek primal = new SocpMosek(instance);
//        System.out.println(intercept+" "+interceptBase);

        SocpGurobi primal = new SocpGurobi(instance, k, reg, diag,constant,terms, false);
        double time = System.currentTimeMillis();
//        double[] xSol = new double[n], zSol = new double[m], obj = new double[4];
        double[] xSol2 = new double[n], zSol2 = new double[m],
                tSol = new double[m], vSol = new double[m], obj2 = new double[4];
        primal.solve(xSol, zSol, vSol, tSol, metrics);
        SocpGurobi solverG = new SocpGurobi(instance, k, regularization, zSol,intercept);
        solverG.solve(xSol2, zSol2, obj2);// Initial primal solution
        metrics[0] = obj2[0];
        metrics[2] += obj2[2];
        if (Double.isInfinite(ubFirst)) {
            ubFirst = metrics[0];
        }
        lb = Math.max(lb, metrics[1]);
        ub = Math.min(ub, metrics[0]);
        double gap = 1, prevGap = Double.POSITIVE_INFINITY;
        gap = (ub - lb) / ub;
        gaps.add(gap);

        double[] diagDir = new double[m];

        //Loop
        while (iter < iterLimit) {
            if (Math.abs(gap - prevGap) < epsilon || iter == 0) { // Increases number of iter if small improvement
                iter++;
            }

//              if(iter<0){
//                System.out.println("ITERATION "+iter);
//                dual.updateSOCPFormulationEq(xSol,zSol);
            dual.updateSDPFormulation(vSol, tSol);
            dual.solve(diagDir);
            for (int i = 0; i < diag.length; i++) {
                diag[i] = (1 / (double) iter) * diagDir[i] + (1 - 1 / (double) iter) * diag[i];
            }

            primal = new SocpGurobi(instance, k, reg, diag,constant,terms, false);

            primal.solve(xSol, zSol, vSol, tSol, metrics);

            solverG = new SocpGurobi(instance, k, regularization, zSol,intercept);
            solverG.solve(xSol2, zSol2, obj2);
            metrics[0] = obj2[0];
            metrics[2] += obj2[2];
            lb = Math.max(lb, metrics[1]);
            ub = Math.min(ub, metrics[0]);
            prevGap = gap;
            gap = (ub - lb) / ub;
            gaps.add(gap);
        }

        System.out.println("");
        System.out.println("UB= " + ub + "\t LB= " + lb);
        System.out.println("Progress");
        for (int i = 0; i < gaps.size(); i++) {
            System.out.println(i + " " + gaps.get(i));
        }
        System.out.println("");

        // Solves the MIP with best parameters found.
        if (mip) {
            System.out.println("MIP");
            double time2 = System.currentTimeMillis();
            time = (time2 - time) / 1000;

            SocpGurobi solver = new SocpGurobi(instance, k, reg, diag,constant,terms, true);
            solver.gurobi.set(GRB.DoubleParam.TimeLimit, 600 - time);

            solver.solve(xSol, zSol, metrics);
            metrics[2] += time;
        }
//        time2=(System.currentTimeMillis()-time2)/1000;

//        SocpMosek solver= new SocpMosek(instance);
//        solver.buildMinimizationFinal(lambda, k);
//
//        solver.solveIP(xSol, zSol, obj);
        System.out.println("LB Relax=" + lb + "\t UB Relax=" + ub + "\t UB last" + ubFirst);

//        System.out.println("LB MIP="+obj[1]+"\t UB MIP="+obj[1]);
    }

}

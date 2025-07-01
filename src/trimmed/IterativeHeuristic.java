/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package trimmed;

import com.gurobi.gurobi.GRBException;
import data.Regression;

/**
 *
 * @author andre
 */
public class IterativeHeuristic {

    double regularization;
    int k;
    Regression instance;
    boolean intercept;

    public IterativeHeuristic(double regularization, int k, boolean intercept, Regression instance) {
        this.regularization = regularization;
        this.k = k;
        this.instance = instance;
        this.intercept=intercept;
    }

    public void solve(double[] xSol, double[] zSol, double[] metrics) throws GRBException {
        int n = instance.n;
        int m = instance.m;

        int iter = 0, iterLimit = 20;
        double lb = 0, ub = Double.POSITIVE_INFINITY, ubLast = ub;

        SocpGurobi solver = new SocpGurobi(instance, k, regularization, 5,true,intercept);

        double[] errors = new double[m], obj = new double[4];
        long time=System.currentTimeMillis();
        solver.solve(xSol, zSol, obj);
        for (int i = 0; i < errors.length; i++) {
            errors[i] = instance.y[i];
            for (int j = 0; j < xSol.length; j++) {
                errors[i] -= instance.A[i][j] * xSol[j];
            }
            errors[i] = Math.abs(errors[i]);
        }

        while (iter < iterLimit) {
            iter++;
            solver = new SocpGurobi(instance, k, regularization, errors,intercept);
            solver.solve(xSol, zSol, obj);
            System.out.println("Obj at iter " + iter + "= " + obj[0] + "\t" + ub);
            ubLast=obj[0];
            ub = Math.min(ub, obj[0]);
            for (int i = 0; i < errors.length; i++) {
                errors[i] = instance.y[i];
                for (int j = 0; j < xSol.length; j++) {
                    errors[i] -= instance.A[i][j] * xSol[j];
                }
                errors[i] = Math.abs(errors[i]);
            }

        }

        metrics[0]=ub;
        metrics[1]=lb;
        metrics[2] = (System.currentTimeMillis() - time) / 1000;
        
        System.out.println("LB Relax=" + lb + "\t UB Relax=" + ub + "\t UB last" + ubLast);

//        System.out.println("LB MIP="+obj[1]+"\t UB MIP="+obj[1]);
    }

}

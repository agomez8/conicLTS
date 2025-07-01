/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package trimmed;

import data.Regression;
import mosek.*;

/**
 * Class for solving the SDP subproblems in the primal dual method. <br>
 * @author Andres Gomez
 */
public class DecompMosekSDP {
    
    /**
     * Minimum diagonal element for the solution (>0 to ensure correctness of a MIO formulation)
     */
    static final double MIN_DIAG=1e-3;

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
    int[] eta, t;

    /**
     * Index of last variable and constraint added to the model
     */
    int varIndex, conIndex;

    /**
     * Total time solving SDPs
     */
    double totalTime;

    //--------------------------------------------------------------------------
    // Constructor
    //--------------------------------------------------------------------------
    /**
     * Constructor.<br>
     *
     * @param instance Regression instance. <br>
     * @param regularization Regularization parameters.
     */
    public DecompMosekSDP(Regression instance, double[][] regularization) {
        env = new Env();
        task = new Task(env, 0, 0);
        this.instance = instance;

        buildCoreSDPFormulation(regularization);
        totalTime = 0;
//        task.putintparam(mosek.iparam.num_threads, 1);
//        task.putdouparam(dparam.mio_max_time, 1800);
//        task.set_Stream(streamtype.log, new Stream() {
//            public void stream(String string) {
//                System.out.println(string);
//            }
//        });

//        buildInstance(a, ub);
    }

    //--------------------------------------------------------------------------
    // Methods
    //--------------------------------------------------------------------------
    /**
     * Builds the feasible region of the SDP formulation. <br>
     *
     * @param lambda Regularization parameter
     *
     */
    private void buildCoreSDPFormulation(double[][] regularization) {
//        task.dispose();
        int n = instance.n;
        int m = instance.m;
        varIndex = 0;
        conIndex = 0;
        t = new int[m];
        eta = new int[m];
        task.appendvars(3 * m); // Epigraph+decomposition+fixed

        // Adds variables
        for (int i = 0; i < m; i++) {
            t[i] = varIndex;
            task.putvarbound(t[i], boundkey.lo, 0, +0.0);
            task.putvarname(t[i], "t" + i);
            task.putcj(t[i], 1);
            varIndex++;

            eta[i] = varIndex;
            task.putvarbound(eta[i], boundkey.lo, 1+MIN_DIAG, 0);
            task.putvarname(eta[i], "eta" + i);
            varIndex++;

//            task.putvarbound(varIndex, boundkey.fx, Math.sqrt(2), Math.sqrt(2));
            task.putvarbound(varIndex, boundkey.fx, 1, 1);
            task.putvarname(varIndex, "one" + i);
            task.appendcone(conetype.rquad, 0, new int[]{t[i], eta[i], varIndex});
            varIndex++;
        }
        
        // PSD Variables
        int barvarindex = 0;
        // /* Append Z. */
        task.appendbarvars(new int[]{n});
        task.appendcons((n) * (n + 1) / 2);

        // Offdiagonal
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double weight = 0.5;
                if (i == j) {
                    weight = 1.0;
                }

                long idx = task.appendsparsesymmat(n,
                        new int[]{j},
                        new int[]{i},
                        new double[]{weight});
                task.putbaraij(conIndex, barvarindex, new long[]{idx}, new double[]{1.0});
                double rhs = regularization[i][j];
                for (int k = 0; k < m; k++) {
                    task.putaij(conIndex, eta[k], instance.A[k][i] * instance.A[k][j]);
                    rhs += instance.A[k][i] * instance.A[k][j];
                }

                task.putconbound(conIndex, boundkey.fx, rhs, rhs);
                task.putconname(conIndex, "con_sdp" + (i) + "," + (j));
                conIndex++;
            }
        }
        barvarindex++;
    }

    /**
     * Updates the formulation with new value of the primal variables. <br>
     * @param w Current value of the continuous variables w. <br>
     * @param tau Current value of the perspective terms w^2/z
     */
    void updateSDPFormulation(double[] w, double[] tau) {
//        task.dispose();
        int m = instance.m;
        
        for (int i = 0; i < m; i++) {
            task.putcj(t[i], tau[i]-w[i]*w[i]);
        }
    }

 
    /**
     * Solves the SDP subproblem. <br>
     * @param diag Array to store the optimal solution. <br>
     * @return The optimal objective value.
     */
    public double solve(double[] diag) {
//        task.writedata("./data.cbf");
        long time = System.currentTimeMillis();
        task.optimize();
        totalTime += (System.currentTimeMillis() - time) / 1000.0;

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
                for (int i = 0; i < diag.length; i++) {
                    diag[i] = 1/solOpt[eta[i]];
                   
                }

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

}

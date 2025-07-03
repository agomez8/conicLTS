/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package trimmed;

import com.gurobi.gurobi.GRBException;
import data.Model_LS;
import data.Regression;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

/**
 * Class for running experiments on real data.
 * @author Andres Gomez
 */
public class TrimmedReal {

    /**
     * Runs different methods for outlier detection on real datasets. <br>
     * @param args Arguments to run the code. <br>
     * args[0]: path to the csv dataset. <br>
     * args[1]: proportion of datapoints to be trimmed as outliers. <br>
     * args[2]: L2 regularization coefficients. <br>
     * args[3]: method. <br>
     * 1= MIO based on big M formulation.
     * 2= MIO based on conic.
     * -300= MIO based on conic+.
     * -4= alternating heuristic.
     * (other parameters like 3-6 may run other methods such as LAD, but are not 
     * used in our experiments with real data).
     * @throws IOException
     * @throws GRBException 
     */
    public static void main(String[] args) throws IOException, GRBException {
        String data = args[0];

        int method = Integer.parseInt(args[3]);
        double proportion = Double.parseDouble(args[1]), gamma = Double.parseDouble(args[2]);

//        int k = Integer.parseInt(args[2]);
        
//        int method = Integer.parseInt(args[3]);
//        int method = 1;

        // Builds linear regression data
        Regression model = new Model_LS(data);
//        model.printOutlierData(data, false, 0);
        int k = (int) (model.m * proportion);

        // Adjust Gamma
        double normColumns = 0;
        for (int j = 0; j < model.n; j++) {
            for (int i = 0; i < model.m; i++) {
                normColumns += model.A[i][j] * model.A[i][j];
            }
        }
        normColumns /= (double) model.n;
        gamma *= normColumns;
        System.out.println("Reg " + gamma);
        if (method > 0) {
            SocpGurobi solverG = new SocpGurobi(model, k, gamma, method,true,false);
            double[] xSol = new double[model.n], zSol = new double[model.m], obj = new double[4];
            solverG.solve(xSol, zSol, obj);
            if (method == 3) {
                for (int i = 0; i < zSol.length; i++) {
                    zSol[i] = 1 - zSol[i];
                }
            } else if (method == 4 || method == 5) {
                for (int i = 0; i < zSol.length; i++) {
                    zSol[i] = 0;
                }
            }

           
            exportSolution(args, obj);

        }
        if (method < 0) {
            SocpMosek solver = new SocpMosek(model);
            switch (method) {
                case -1:
                    {
                        solver.buildMinimizationZioutas(gamma, k);
                        double[] xSol = new double[model.n], zSol = new double[model.m], obj = new double[4];
                        solver.solveIP(xSol, zSol, null,obj);
                        exportSolution(args, obj);
                        break;
                    }
                case -2:
                    {
                        double[][] lambda = new double[model.m][model.n];
                        for (int i = 0; i < lambda.length; i++) {
                            for (int j = 0; j < lambda[i].length; j++) {
                                lambda[i][j] = gamma / (double) model.m;
                                
                            }
                        }       solver.buildMinimizationFinal(lambda, k);
                        double[] xSol = new double[model.n ], zSol = new double[model.m],
                                tauSol=new double[model.m],obj = new double[4];
                        solver.solveIP(xSol, zSol,tauSol, obj);
                        exportSolution(args, obj);
                        break;
                    }
                case -3:
                case -300:
                case -301:
                    {
                        PrimalDualSDP heur = new PrimalDualSDP(gamma, k, model, method!=-301,false,0);
                        double[] xSol = new double[model.n ], zSol = new double[model.m], obj = new double[4];
                        heur.solve(xSol, zSol, obj);
                        exportSolution(args, obj);
                        break;
                    }
                case -4:
                    {
                        IterativeHeuristic heur = new IterativeHeuristic(gamma, k,false, model);
                        double[] xSol = new double[model.n ], zSol = new double[model.m], obj = new double[4];
                        heur.solve(xSol, zSol, obj);
                        exportSolution(args, obj);
                        break;
                    }
                default:
                    break;
            }
        }
    }

  
    /**
     * Exports the solution to a CSV file. <br>
     * @param args Arguments of the instance used. <br>
     * @param vals Values to export. <br>
     * @throws IOException 
     */
    static void exportSolution(String[] args, double[] vals) throws IOException {

        try ( FileWriter out = new FileWriter(new File("./results/resultsLTSReal.csv"), true)) {
            for (int i = 0; i < args.length; i++) {
                out.write(args[i] + ", ");
            }

            for (int i = 0; i < vals.length; i++) {
                out.write(vals[i] + ", ");
            }
           
            out.write("\n");

        }
    }
}

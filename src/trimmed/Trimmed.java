/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package trimmed;

import com.gurobi.gurobi.GRBException;
import data.Model_SyntLS;
import data.Regression;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

/**
 * Solves LTS problems with synthetic data. <br>
 *
 * @author Andres Gomez
 */
public class Trimmed {

    /**
     *
     * @param args Arguments to be used. <br>
     * args[0]: Number of features. <br>
     * args[1]: Number of datapoints. <br>
     * args[2]: Pattern to generate outliers: we use 1 by default in the paper.
     * <br>
     * args[3]: Proportion of outliers. <br>
     * args[4]: L2 regularization parameter. <br>
     * args[5]: Seed. <br>
     * args[6]: Method: 1= bigM. 2= Conic. -300= Conic+. 3: Least median of
     * squares from Bertsimas and Mazumder. 4: Least absolute deviations. 5:
     * Ridge regression. 6: Huber loss.
     * @throws IOException
     * @throws GRBException
     */
    public static void main(String[] args) throws IOException, GRBException {
//        String data = args[0];
//        String data = "./data/BreastCancer.csv";
        int n = Integer.parseInt(args[0]), m = Integer.parseInt(args[1]), pattern = Integer.parseInt(args[2]),
                seed = Integer.parseInt(args[5]), method = Integer.parseInt(args[6]);
        double proportion = Double.parseDouble(args[3]), gamma = Double.parseDouble(args[4]);

        int k = (int) (m * proportion);

        boolean intercept = true;
        Regression model;
        if (intercept) {
            model = new Model_SyntLS(n, m, n - 1, pattern, proportion, 0, seed);
        } else {
            model = new Model_SyntLS(n, m, n, pattern, proportion, 0, seed);
        }
        double[] truth = ((Model_SyntLS) model).trueBeta;

        if (method > 0) {
            SocpGurobi solverG = new SocpGurobi(model, k, gamma, method, true, intercept);
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

            xSol = ((Model_SyntLS) model).convert(xSol);
            System.out.print("Truth\t");
            for (double d : truth) {
                System.out.print(d + " ");
            }
            System.out.println("");
            System.out.print("Estimation\t");
            for (double d : xSol) {
                System.out.print(d + " ");
            }
            System.out.println("");

//            stats[0] = model.computeMetrics(xSol)[1];
            double[] stats2 = computeStatMetrics(xSol, zSol, k, truth);
            double[] stats = model.computeMetrics(xSol);
            stats[5] = stats2[1];
            stats[6] = stats2[2];
            exportSolution(args, obj, stats, xSol, truth);

        }
        if (method < 0) {
            SocpMosek solver = new SocpMosek(model);

            if (method == -3 || method == -300 || method == -301) {
                double baseIntercept = 0;
                if (intercept) {
                    IterativeHeuristic heur0 = new IterativeHeuristic(gamma, k, intercept, model);
                    double[] xSol = new double[model.n], zSol = new double[model.m], obj = new double[4];
                    heur0.solve(xSol, zSol, obj);
                    baseIntercept = xSol[xSol.length - 1];
                }
                PrimalDualSDP heur = new PrimalDualSDP(gamma, k, model, method != -301, intercept, baseIntercept);
                double[] xSol = new double[model.n], zSol = new double[model.m], obj = new double[4];
                heur.solve(xSol, zSol, obj);

                xSol = ((Model_SyntLS) model).convert(xSol);

                System.out.print("Truth\t");
                for (double d : truth) {
                    System.out.print(d + " ");
                }
                System.out.println("");
                System.out.print("Estimation\t");
                for (double d : xSol) {
                    System.out.print(d + " ");
                }
                System.out.println("");

//                double[] stats = computeStatMetrics(xSol, zSol, k, truth);
//                stats[0] = model.computeMetrics(xSol)[1];
                double[] stats2 = computeStatMetrics(xSol, zSol, k, truth);
                double[] stats = model.computeMetrics(xSol);
                stats[5] = stats2[1];
                stats[6] = stats2[2];
                exportSolution(args, obj, stats, xSol, truth);
            } else if (method == -4) {
                IterativeHeuristic heur = new IterativeHeuristic(gamma, k, intercept, model);
                double[] xSol = new double[model.n], zSol = new double[model.m], obj = new double[4];
                heur.solve(xSol, zSol, obj);

                xSol = ((Model_SyntLS) model).convert(xSol);
                System.out.print("Truth\t");
                for (double d : truth) {
                    System.out.print(d + " ");
                }
                System.out.println("");
                System.out.print("Estimation\t");
                for (double d : xSol) {
                    System.out.print(d + " ");
                }
                System.out.println("");

//                double[] stats = computeStatMetrics(xSol, zSol, k, truth);
//                stats[0] = model.computeMetrics(xSol)[1];
                double[] stats2 = computeStatMetrics(xSol, zSol, k, truth);
                double[] stats = model.computeMetrics(xSol);
                stats[5] = stats2[1];
                stats[6] = stats2[2];
                exportSolution(args, obj, stats, xSol, truth);
            }
        }
    }

    /**
     * Computes metrics associated with the quality of the solution. <br>
     * @param xSol Value of the regression coefficients. <br>
     * @param zSol Value of indicator variables for outliers. <br>
     * @param outliers Number of outliers in the data. <br>
     * @param truth Vector with the ground truth. <br>
     * @return Array with the metrics for the solution quality:
     * metrics[0]= Relative error.
     * metrics[1]= Number of false negatives
     * metrics[2]= Number of false positives.
     */
    static double[] computeStatMetrics(double[] xSol, double[] zSol, int outliers, double[] truth) {
        double[] metrics = new double[3];
        for (int i = 0; i < zSol.length; i++) {
            if (zSol[i] < 0.5 & i < outliers) {
                metrics[1]++;
            } else if (zSol[i] > 0.5 & i >= outliers) {
                metrics[2]++;
            }
        }
        double norm2 = 0, error = 0;
        for (int i = 0; i < truth.length; i++) {
            norm2 += truth[i] * truth[i];
            error += (truth[i] - xSol[i]) * (truth[i] - xSol[i]);
        }
        metrics[0] = error / norm2;
        return metrics;
    }

    /**
     * Exports the results to a CSV file. <br>
     * @param args Arguments of the instance used. <br>
     * @param vals Optimization metrics. <br>
     * @param stats Statistical metrics. <br>
     * @param sol Regression coefficients found. <br>
     * @param truth Ground truth. <br>
     * @throws IOException 
     */
    static void exportSolution(String[] args, double[] vals, double[] stats, double[] sol, double[] truth) throws IOException {

        try (FileWriter out = new FileWriter(new File("./results/resultsTrimmedHuber.csv"), true)) {
            for (int i = 0; i < args.length; i++) {
                out.write(args[i] + ", ");
            }

            for (int i = 0; i < vals.length; i++) {
                out.write(vals[i] + ", ");
            }
            for (int i = 0; i < stats.length; i++) {
                out.write(stats[i] + ", ");
            }
            out.write(", ");
            for (int i = 0; i < sol.length; i++) {
                out.write(sol[i] + ", ");
            }
            out.write(", ");
            for (int i = 0; i < truth.length; i++) {
                out.write(truth[i] + ", ");
            }
            out.write("\n");

        }
    }
}

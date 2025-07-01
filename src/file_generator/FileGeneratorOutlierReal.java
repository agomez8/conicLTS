/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package file_generator;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

/**
 *
 * @author Andres Gomez.
 */
public class FileGeneratorOutlierReal {

    public static void main(String[] args) throws IOException {

        String instance = "java "
                //                + "-Djava.library.path=\"C:/Program Files/IBM/ILOG/CPLEX_Studio1271/cplex/bin/x64_win64" 
                + " -cp ./dist/Linear_Regression.jar trimmed.TrimmedReal";

        String[] datasets = {//"./data_robust/aircraft.csv",
            "./data_robust/alcohol.csv", "./data_robust/bushfire.csv", "./data_robust/cloud.csv",
            "./data_robust/coleman.csv", "./data_robust/delivery.csv", "./data_robust/education.csv",
            "./data_robust/epilepsy.csv", "./data_robust/foodstamp.csv", "./data_robust/hbk.csv",
            "./data_robust/heart.csv", "./data_robust/lactic.csv", "./data_robust/milk.csv",
//            "./data_robust/NoxEmissions.csv", 
            "./data_robust/pension.csv", "./data_robust/phosphor.csv",
            "./data_robust/pilot.csv", "./data_robust/pulpfiber.csv", "./data_robust/radarImage.csv",
            "./data_robust/salinity.csv", "./data_robust/starsCYG.csv", "./data_robust/steamUse.csv",
            "./data_robust/toxicity.csv", "./data_robust/wagnerGrowth.csv", "./data_robust/wood.csv"};

//        String[] datasets = { "./data/Crime.csv"};
        double[] proportions = {0.1, 0.2,0.3, 0.4};
        double[] lambdas = {0.05, 0.1, 0.2};
        int[] methods = { -4,-301,1, 2,-300};

        try ( FileWriter out = new FileWriter(new File("./runOutlierReal.bat"))) {
            for (String dataset : datasets) {
                for (double proportion : proportions) {
                    for (double lambda : lambdas) {

                        for (int method : methods) {

                            out.write(instance + " " + dataset + " "
                                    + proportion + " " + lambda + " " + method + "\n");

                        }

                    }
                }
            }
        }

    }
}

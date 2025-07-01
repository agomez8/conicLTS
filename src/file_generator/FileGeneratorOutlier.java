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
public class FileGeneratorOutlier {

    public static void main(String[] args) throws IOException {

        String instance = "java "
                //                + "-Djava.library.path=\"C:/Program Files/IBM/ILOG/CPLEX_Studio1271/cplex/bin/x64_win64" 
                + " -cp ./dist/Linear_Regression.jar trimmed.Trimmed";

//        String[] datasets = {"./data/Housing.csv", "./data/Servo.csv", "./data/AutoMPG.csv",
//            "./data/FlaresC.csv", "./data/BreastCancer.csv",
//            "./data/diabetesQ.csv"};
        int[] ns = new int[]{3,21};
        int[] ms = new int[]{100,500};
        int[] patterns = new int[]{1};
//        int[] patterns = new int[]{1,23};
        double[] props = new double[]{0.1, 0.2, 0.4};
//        double[] lambdas = {0.01, 0.1, 0.2, 0.3};
        double[] lambdas = {0.01, 0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
        int[] seeds = {101, 102, 103, 104, 105};
//        int[] methods = {-4, -3, -2, -1, 1, 2, 3, 4, 5};
//        int[] methods = {3,4,5,-4,-300,-301,1,2};
        int[] methods = {6};
        try ( FileWriter out = new FileWriter(new File("./runOutlierLarge.bat"))) {
            for (int n : ns) {
                for (int m : ms) {
                    for (int pattern : patterns) {

                        for (double prop : props) {
                            for (double lambda : lambdas) {
                                for (int seed : seeds) {
                                    for (int method : methods) {
                                        out.write(instance + " " + n + " " + m + " "
                                                + pattern + " " + prop + " " + lambda
                                                + " " + seed + " " + method + "\n");
                                    }
                                }
                            }
                        }
                    }

                }

            }
        }
    }
}

[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# ConicLTS


This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The software and data in this repository are a snapshot of the software and data
that were used in the research reported on in the paper 
[Outlier detection in regression: Conic quadratic formulations](https://doi.org/10.1287/ijoc.2025.1215) by Andres Gomez and Jose Neto. 


**Important: This code is being developed on an on-going basis at 
https://github.com/agomez8/conicLTS. Please go there if you would like to
get a more recent version or would like support**

## Cite

To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

https://doi.org/10.1287/ijoc.2025.1215

https://doi.org/10.1287/ijoc.2025.1215.cd

Below is the BibTex for citing this snapshot of the repository.

```
@misc{ConicLTS,
  author =        {Andres Gomez and Jose Neto},
  publisher =     {INFORMS Journal on Computing},
  title =         {{ConicLTS}},
  year =          {2025},
  doi =           {10.1287/ijoc.2025.1215.cd},
  url =           {https://github.com/INFORMSJoC/2025.1215},
  note =          {Available for download at https://github.com/INFORMSJoC/2025.1215},
}  
```

## Description

The goal of this software is to demonstrate the use of mixed-integer conic quadratic formulations to detect outliers in regression problems (closely related to the Least Trimmed Squares problem in statistics).

The methods are implemented in Java and rely on commercial solvers Gurobi and Mosek. Executing the code requires a license for these solvers.

## Executing the code

As a java code, the source code is precompiled and can be executed directly via file ./dist/LTS.jar. Ensure to install Gurobi and Mosek and replace files gurobi.jar and mosek.jar in ./dist/lib with those obtaining from installing these software.

The code can be executed from the console. Files "runOutlierSynt.bat" and "runOutlierReal.bat" contain example of how to execute the code to solve synthetic and real instances. 

An example command to execute the code to tackle a synthetic instance is
```
java  -cp ./dist/LTS.jar trimmed.Trimmed 3 100 1 0.1 0.01 101 3
```
where: "java  -cp ./dist/LTS.jar" points to the direction of the executable jar file, and "trimmed.Trimmed" is the class used to run synthetic instances. The rest of parameters are as follows:
* First parameter (3) is the number of features
* Second parameter (100) is the number of datapoints
* Third parameter (1) controls the synthetic data generation; 1 is used in the paper.
* Fourth parameter (0.1) is the proportion of datapoints that should be discarded as outliers.
* Fifth parameter (0.01) is the value of the L2 regularization
* Sixth parameter (101) is the seed
* Seventh parameter (3) is the method to be used. Method: 1= bigM. 2= Conic. -300= Conic+. 3: Least median of
 squares from Bertsimas and Mazumder (2014). 4: Least absolute deviations. 5: Ridge regression. 6: Huber loss.

An example command to execute the code to tackle a real instance is
```
java  -cp ./dist/LTS.jar trimmed.TrimmedReal ./data_robust/alcohol.csv 0.1 0.05 -4
```
where: "java  -cp ./dist/Linear_Regression.jar" points to the direction of the executable jar file, and "trimmed.TrimmedReal" is the class used to run real instances. The rest of parameters are as follows:
* First parameter (./data_robust/alcohol.csv) is the path to the dataset in csv format
* Second parameter(0.1) is the proportion of datapoints that should be discarded as outliers.
* Third parameter (0.05) is the value of the L2 regularization
* Fourth parameter (-4) is the method to be used. Method: 1= MIO based on big M formulation. 2= MIO based on conic. -300= MIO based on conic+. -4= alternating heuristic.

## Output

After solving an instance, the results are recorded in "./results/resultsLTSSynt.csv" (for synthetic instances) and "./results/resultsLTSReal.csv" (for real instances). Each instance solved is added as a new row to these files. 

For synthetic data, each row is organized as follows:
* Columns 1-7:  are the parameters used to generate the instance.
* Column 8: best objective value found.
* Column 9: best bound found.
* Column 10: time in seconds.
* Column 11: number branch-and-bound nodes.
* Column 8: training error.
* Column 9: prediction error.
* Column 10: relative risk.
* Column 11: relative test error.
* Column 12: proportion of variance explained.
* Column 13: number of false negatives.
* Column 14: number of false positives.
* Column 16+: regression coefficients obtained and ground truth

For real data, each row is organized as follows:
* Columns 1-4:  are the parameters used to generate the instance.
* Column 5: best objective value found.
* Column 6: best bound found.
* Column 7: time in seconds.
* Column 8: number branch-and-bound nodes.


In Linux, to build the version that multiplies all elements of a vector by a
constant (used to obtain the results in [Figure 1](results/mult-test.png) in the
paper), stepping K elements at a time, execute the following commands.

```
make mult
```

Alternatively, to build the version that sums the elements of a vector (used
to obtain the results [Figure 2](results/sum-test.png) in the paper), stepping K
elements at a time, do the following.

```
make clean
make sum
```

Be sure to make clean before building a different version of the code.

## Results

Figure 1 in the paper shows the results of the multiplication test with different
values of K using `gcc` 7.5 on an Ubuntu Linux box.

![Figure 1](results/mult-test.png)

Figure 2 in the paper shows the results of the sum test with different
values of K using `gcc` 7.5 on an Ubuntu Linux box.

![Figure 1](results/sum-test.png)

## Replicating

To replicate the results in [Figure 1](results/mult-test), do either

```
make mult-test
```
or
```
python test.py mult
```
To replicate the results in [Figure 2](results/sum-test), do either

```
make sum-test
```
or
```
python test.py sum
```

## Ongoing Development

This code is being developed on an on-going basis at the author's
[Github site](https://github.com/tkralphs/JoCTemplate).

## Support

For support in using this software, submit an
[issue](https://github.com/tkralphs/JoCTemplate/issues/new).

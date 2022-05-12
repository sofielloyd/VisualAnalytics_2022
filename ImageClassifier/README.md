# Image classifier benchmark scripts - Assignment 2
This repository contains all of the code and data related to the second assignment for my portfolio exam in the Spring 2022 module **Visual Analytics** which is a part 
of my tilvalg in Cultural Data Science at Aarhus University.  

This repository is in active development, with new material being pushed on regularly from now and until **19th may 2022**.


## Assignment description 
Take the classifier pipelines we covered in lecture 7 and turn them into *two separate ```.py``` scripts*.  

My ```.py```script does the following:

- Make one script should be called ```logistic_regression.py```.
  - Load either the **MNIST_784** data or the **CIFAR_10** data
  - Train a Logistic Regression model using ```scikit-learn```.
  - Print the classification report to the terminal **and** save the classification report to ```output/lr_report.txt```.  
 
- Make another scripts should be called ```nn_classifier.py```. 
  - Load either the **MNIST_784** data or the **CIFAR_10** data.
  - Train a Neural Network model using the premade module in ```neuralnetwork.py```.
  - Print output to the terminal during training showing epochs and loss.
  - Print the classification report to the terminal **and** save the classification report to ```output/nn_report.txt```.


## Goal and outcome of the assignment
- The goal of this assignment is to demonstrate that I know how to create ```.py``` scripts with simple classifiers that can act as benchmarks for future reasearch.
- The code will provide a couple of scripts which could be re-written and reused on separate data.


## Methods  



## Usage    
In order to reproduce this code, you'll need to uploade your own data into the ```input``` folder.  
I have used the **CIFAR_10** dataset which can be loaded from the ```tensorflow``` package.    

You'll also have to install the needed packages, which can be found in ```requirements.txt```.    

The scripts can be run from the command line by changing the directory to ```ImageClassifier``` and then execute ```python src/logistic_regression.py``` *or* ```python src/nn_classifier.py```.  


### Repo Structure  
This repository has the following directory structure:  

| **Folder** | **Description** |
| ----------- | ----------- |
| ```input``` | Input data |
| ```output``` | Output data |
| ```src``` | Python scripts |
| ```utils``` | Additional Python functions |


- The ```input``` folders are empty and this is where you should upload your own data, if you want to reproduce the code. **CHANGE THIS** *The input data that I have used for the given code is described in the specific assignmentfolder.*

- The ```output``` folders contains my results and it is this folder that you should save you own results when replicating the code. 

- The ```src``` folders contains the code written in ```.py``` scripts. 

- The ```utils``` folders contains a collection of small Python functions which make common patterns shorter and easier. The utils scripts used in this project were developed in-class and can also be found in [this](https://github.com/CDS-AU-DK/cds-visual.git) repository.


## Discussion of results 
### Results for ```logistic_regression.py```  
- The output of this script is the ```lr_report.txt``` which can be found in the output folder.   
- The accuracy score is 0.31 which is rather low. 
- The precision score is best for ```truck``` and worst for ```cat```.  
- Overall the precision score is highest on machines and lowest on animals. 

### Results for ```nn_classifier.py```
- The output of this script is the ```nn_report.txt``` which can be found in the output folder. 
- The accuracy score is 0.37 which low and not that much better than the logistic regression. 
- The precision score is best for ```airplane``` and worst for ```cat```. 
- Overall the precision score is highest on machines and lowest on animals. 


## Further development 
Improvements for this code could be to: 
- Add ```argparse()``` so that the scripts use either **MNIST_784** or **CIFAR_10** based on some input from the user on the command line.
- Add ```argparse()``` to allow users to define the number and size of the layers in the neural network classifier.

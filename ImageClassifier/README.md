# Image classifier benchmark scripts - Assignment 2
This repository contains all of the code and data related to the second assignment for my portfolio exam in the Spring 2022 module **Visual Analytics** which is a part 
of my tilvalg in Cultural Data Science at Aarhus University.  

This repository is in active development, with new material being pushed on regularly from now and until **19th may 2022**.


## Assignment description 
Take the classifier pipelines we covered in lecture 7 and turn them into *two separate ```.py``` scripts*.  

My ```.py```script does the following:

- Make one script should be called ```logistic_regression.py```.
  - Load either the **MNIST_784** data or the **CIFAR_10** data.
  - Train a Logistic Regression model using ```scikit-learn```.
  - Print the classification report to the terminal **and** save the classification report to ```output/lr_report.txt```.  
 
- Make another scripts should be called ```nn_classifier.py```. 
  - Load either the **MNIST_784** data or the **CIFAR_10** data.
  - Train a Neural Network model using the premade module in ```neuralnetwork.py```.
  - Print output to the terminal during training showing epochs and loss.
  - Print the classification report to the terminal **and** save the classification report to ```output/nn_report.txt```.


## Goal and outcome of the assignment
- The goal of this assignment is to demonstrate that I know how to create ```.py``` scripts with simple classifiers that can act as benchmarks for future reasearch.
- - The code will provide a couple of scripts which could be re-written and reused on separate data.


## Methods  
*This section may contain the same as in the "Goals and outcome of the assignment" section? Maybe I should delete that section and re-write it into this section?* 


## Usage    
In order to reproduce this code, you'll need to uploade your own data into the ```input``` folder.
The script can be run from the command line by changing the directory to ```ImageClassifier``` and then execute ```python src/logistic_regression.py``` *or* ```python src/nn_classifier.py```. 

**Follow up on this** *You'll also have to  install the dependencies from a requirements.txt*


### Repo Structure  
This repository has the following directory structure:  

| **Folder** | **Description** |
| ----------- | ----------- |
| ```input``` | Input data |
| ```output``` | Output data |
| ```src``` | Python scripts |
| ```utils``` | Additional Python functions |


- The ```input``` folders are empty and this is where you should upload your own data, if you want to reproduce the code. The input data that I have used for the given code is described in the specific assignmentfolder.

- The ```output``` folders contains my results and it is this folder that you should save you own results when replicating the code. 

- The ```src``` folders contains the code written in ```.py``` scripts. 

- The ```utils``` folders contains a collection of small Python functions which make common patterns shorter and easier. 


## Discussion of results 
*What does my outputs show? What is the results?* 


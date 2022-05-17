# Yoga poses CNN classification - Self-assigned project
This repository contains all of the code and data related to my self-assigned project for my portfolio exam in the Spring 2022 module **Visual Analytics** which is a part of my tilvalg in Cultural Data Science at Aarhus University.  

## Assignment description 
My ```.py```script does the following:
- Load the **Yoga Poses Dataset**.
- Train a CNN classifier using ```tensorflow```.
- Save the sequential model.
- Save plots of the loss and accuracy.
- Print the classification report to the terminal **and** save the classification report.


### Goal and outcome of the assignment
- The goal of the self assigned project is to demonstrate my skills in Python related to image processing. 
- The code will provide a script which could be re-written and reused on separate data.


## Methods  

## Usage    
In order to reproduce this code, you'll need to uploade your own data into the ```input``` folder.  
I have used the **Yoga Poses Dataset** found on Kaggle. The dataset can be found [here](https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset).   
However, I have merged the *TEST* and *TRAIN* directory into one directory called *dataset*. 

You'll also have to install the needed packages, which can be found in ```requirements.txt```.    

The script can be run from the command line by changing the directory to ```src``` and then execute ```python yoga_poses.py```.


### Repo Structure  
This repository has the following directory structure:  

| **Folder** | **Description** |
| ----------- | ----------- |
| ```input``` | Input data |
| ```output``` | Output data |
| ```src``` | Python scripts |


- The ```input``` folders are empty and this is where you should upload your own data, if you want to reproduce the code.

- The ```output``` folders contains my results and it is this folder that you should save you own results when replicating the code. 

- The ```src``` folders contains the code written in ```.py``` scripts. 


## Discussion of results 
### Results for ```yoga_poses.py```  
- The final train accuracy was 0.97 and the final validation accuracy was 0.73
- The output of this script is the ```report.txt``` which can be found in the output folder.   
- The accuracy from the classification report was 0.24.
- The precision score is best for ```warrior2``` which had a precision score on 0.29, and worst for ```tree``` which had a precision score on 0.17.  


## Further development 
Improvements for this code could be to: 
- Use argparse() to allow users to define specific hyperparameters in the script.
  - This might include e.g. learning rate, batch size, etc.
- The user should be able to define the names of the output plot and output classification report from the command line.

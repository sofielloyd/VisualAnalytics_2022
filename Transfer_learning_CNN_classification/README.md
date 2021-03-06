# Transfer Learning & CNN classification - Assignment 3
This repository contains all of the code and data related to the third assignment for my portfolio exam in the Spring 2022 module **Visual Analytics** which is a part of my tilvalg in Cultural Data Science at Aarhus University.  


## Assignment description 
In this assignment, I'll be working with the ```CIFAR10``` dataset and build a classifier using transfer learning with a pretrained CNN like VGG16 for feature extraction. 

My ```.py``` script does the following:

- Load the CIFAR10 dataset
- Use VGG16 to perform feature extraction
- Train a classifier 
- Save plots of the loss and accuracy 
- Save the classification report


### Goals and outcome of the assignment
- The purpose of this assignment is to show that I am able to use transfer learning in the context of image data, a state-of-the-art task in deep learning.
- This assignment is also intended to increase my familiarity of working with ```Tensorflow/Keras```, and with building complex deep learning pipelines.
- The code will provide a basic outline of a classification pipeline using transfer learning which can be adapted and modified for future tasks.  


## Methods  
This code provides a deep learning method using ```Tensorflow``` to perform transfer learning and classifing color images from the CIFAR10 dataset.    
I first used a pre-trained CNN to extract features using VGG16.   
I then used ```Scikit-Learn``` to print the classification report to the terminal.


## Usage    
In order to reproduce this code, you'll need to uploade your own data into the ```input``` folder.
I have used the **CIFAR_10** dataset which can be loaded from the ```tensorflow``` package. 

You'll also have to install the needed packages, which can be found in ```requirements.txt```.  

The script can be run from the command line by changing the directory to ```src``` and then execute ```python transfer_learning.py```.


### Repo Structure  
This repository has the following directory structure:  

| **Folder** | **Description** |
| ----------- | ----------- |
| ```input``` | Input data |
| ```output``` | Output data |
| ```src``` | Python scripts |


- The ```input``` folders are empty and this is where you should upload your own data, if you want to reproduce the code.

- The ```output``` folders contains my results and it is this folder that you should save your own results when replicating the code. 

- The ```src``` folders contains the code written in ```.py``` scripts. 


## Discussion of results 
### Results
- The output of this script is the ```report.txt``` which can be found in the output folder.
- The results of my script gave an accuracy score on 0.57. 
- It also showed that the label ```ship``` had the highest precision score on 0.76 and the label ```bird``` had the lowest precision score on 0.41. 
- Overall the precision score is highest on machines and lowest on animals.

### Further development 
Improvements for this code could be to: 
- Use argparse() to allow users to define specific hyperparameters in the script.
  - This might include e.g. learning rate, batch size, etc.
- The user should be able to define the names of the output plot and output classification report from the command line.

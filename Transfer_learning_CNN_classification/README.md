# Transfer Learning & CNN classification - Assignment 3
This repository contains all of the code and data related to the third assignment for my portfolio exam in the Spring 2022 module **Visual Analytics** which is a part 
of my tilvalg in Cultural Data Science at Aarhus University.  

This repository is in active development, with new material being pushed on regularly from now and until **19th may 2022**.


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
This code provides a deep learning method using ```Tensorflow``` to perform transfer learning and classifing color images from the CIFAR10 dataset.  I first used   a pre-trained CNN to extract features using VGG16. I also uses ```Scikit-Learn``` to print the classification report to the terminal.


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


- The ```input``` folders are empty and this is where you should upload your own data, if you want to reproduce the code. **CHANGE THIS** *The input data that I have used for the given code is described in the specific assignmentfolder.*

- The ```output``` folders contains my results and it is this folder that you should save you own results when replicating the code. 

- The ```src``` folders contains the code written in ```.py``` scripts. 


## Discussion of results 
*What does my outputs show? What is the results?* 

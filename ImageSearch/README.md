# Image Search - Assignment 1
This repository contains all of the code and data related to the first assignment for my portfolio exam in the Spring 2022 module **Visual Analytics** which is a part 
of my tilvalg in Cultural Data Science at Aarhus University.  

This repository is in active development, with new material being pushed on regularly from now and until **19th may 2022**.


## Assignment description 
Write a small Python program to compare image histograms quantitively using ```Open-CV``` and the other image processing tools.

My ```.py```script should do the following:
- Take a user-defined image from the folder.
- Calculate the "distance" between the colour histogram of that image and all of the others.
- Find which 3 image are most "similar" to the target image.
- Save an image which shows the target image, the three most similar, and the calculated distance score.
- Save a CSV which has one column for the filename and three columns showing the filenames of the closest images in descending order.


### Goal and outcome of the assignment 
- The goal of this assignment is to demonstrate that I have a good understanding of how to use simple image processing techniques to extract valuable information 
from image data.
- The code will provide a simple tool for performing image search on a dataset of images, finding which images are most similar to one another


## Methods  
*This section may contain the same as in the "Goals and outcome of the assignment" section? Maybe I should delete that section and re-write it into this section?* 


## Usage
In order to reproduce this code, you'll need to uploade your own data into the ```input``` folder.
The script can be run from the command line by changing the directory to ```ImageSearch``` and then execute ```python src/ImageSearch.py```.

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

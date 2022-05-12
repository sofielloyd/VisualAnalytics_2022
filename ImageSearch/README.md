# Image Search - Assignment 1
This repository contains all of the code and data related to the first assignment for my portfolio exam in the Spring 2022 module **Visual Analytics** which is a part of my tilvalg in Cultural Data Science at Aarhus University.  


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
- The code will provide a simple tool for performing image search on a dataset of images, finding which images are most similar to one another.


## Methods  



## Usage
In order to reproduce this code, you'll need to uploade your own data into the ```input``` folder. 
I have used the **flower dataset** which can be found [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).  

You'll also have to install the needed packages, which can be found in ``requirements.txt```. 

I have made two scripts for this assignment. 
1. The first one is the ```compare_images.py``` which is a simple script using ```Open-CV``` to compare the color histogram of three images.  
This script can be run from the command line by changing the directory to ```ImageSearch``` and then execute ```python src/compare_images.py```.   

2. The second script is ```image_search.py``` which is a script that makes image embedding using ```NearestNeighbors``` from ```scikit-learn```. 
I have added a parser for this script and made it possible for the user to enter either a *single filename* or a *directory* on the command line. This script can be run from the command line by changing the directory to ```ImageSearch``` and then execute ```python src/ImageSearch.py -fn *filename*``` for running the script on a single file or ```python src/ImageSearch.py -d "../input"``` for running the script on the whole input folder.  
 

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

- The ```utils``` folders contains a collection of small Python functions which make common patterns shorter and easier. 


## Discussion of results 
**Results for ```compare_images.py```:**
- The output of this script is the results of the three scores.   

**Results for ```ìmage_search.py```:**
- If the user runs the code for a single file, the output of this script will be a ```.csv``` file with the filename of the targetimage and the three most similar images, and two images; one with the target image and one with the three most similar images.  
- If the user runs the code for the whole directory, the output of this script will be a ```.csv``` file with every filename and the three most similar images to every images. These results will be saved to the ```output``` folder. 

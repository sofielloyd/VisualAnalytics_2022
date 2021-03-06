# Image Search - Assignment 1
This repository contains all of the code and data related to the first assignment for my portfolio exam in the Spring 2022 module **Visual Analytics** which is a part of my tilvalg in Cultural Data Science at Aarhus University.  


## Assignment description 
Write a small Python program to compare image histograms quantitively using ```Open-CV``` and the other image processing tools.

My ```.py```scripts does the following:
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
### ```compare_images.py```
- The ```compare_images.py``` uses ```Open-CV``` to compare the color histogram of three images. 
- I have used the ```cv2.HISTCMP_CHISQR``` function that calculates the Chi-Squared distance between two histograms.

### ```image_search.py```
- The ```image_search.py``` makes image embeddings using ```NearestNeighbors``` from ```scikit-learn```. 
- I have used the ```distances, indices = neighbors.kneighbors()``` which calulates the cosine distance to find the image which are closest to the target image.  
- I have added a parser for this script and made it possible for the user to enter either a *single filename* or a *directory* on the command line.      


## Usage
In order to reproduce this code, you'll need to uploade your own data into the ```input``` folder.   
I have used the **flower dataset** which can be found [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html).  

You'll also have to install the needed packages, which can be found in ```requirements.txt```. 

The ```compare_images.py``` can be run from the command line by changing the directory to ```src``` and then execute ```python compare_images.py```.   

The ```image_search.py``` can be run from the command line by changing the directory to ```src``` and then execute ```python ImageSearch.py -fn *filename*``` for running the script on a single file or ```python ImageSearch.py -d *path to directory*``` for running the script on a directory.   
If you use the input folder, then the path to directory should be  ```"../input"```.
 

### Repo Structure  
This repository has the following directory structure:  

| **Folder** | **Description** |
| ----------- | ----------- |
| ```input``` | Input data |
| ```output``` | Output data |
| ```src``` | Python scripts |
| ```utils``` | Additional Python functions |


- The ```input``` folders are empty and this is where you should upload your own data, if you want to reproduce the code.

- The ```output``` folders contains my results and it is this folder that you should save you own results when replicating the code. 

- The ```src``` folders contains the code written in ```.py``` scripts. 

- The ```utils``` folders contains a collection of small Python functions which make common patterns shorter and easier. The utils scripts used in this project were developed in-class and can also be found in [this](https://github.com/CDS-AU-DK/cds-visual.git) repository.


## Discussion of results 
### Results for ```compare_images.py```
- The output of this script is the results of the three distance scores.   

#### Futher development  
- This code could be improved by adding ```argparse()``` so that user can chose from the command line which images should be used for the comparison. 


### Results for ```image_search.py```
- If the user runs the code for a single file, the output of this script will be a ```.csv``` file with the filename of the target image and the three most similar images, and two images; one with the target image with the distance score and one with the three most similar images.    
The filenames for these output files is ```*filename*_similar_images.csv```, ```*filename*_similar_images.png``` and ```*filename*_target_image.png```.

- If the user runs the code for the whole directory, the output of this script will be a ```.csv``` file with every filename and the three most similar images to every images.   
The file is called ```similar_images.csv```.
- These results will be saved to the ```output``` folder.  

#### Further development 
- An improvement of this code could be to save the images as one image with the target image and the three similar images.  
- Another improvement could be to save the distance score from the ```kneighbors``` function in the ```similar_images.csv``` with the filenames, like this: 

| **Target_image** | **First_similar** | **First_similar_distance** | **Second_similar** | **Second_similar_distance** | **Third_similar** | **Third_similar_distance** |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | 
| filename.jpg | filename.jpg | distance score |  filename.jpg | distance score | filename.jpg | distance score |

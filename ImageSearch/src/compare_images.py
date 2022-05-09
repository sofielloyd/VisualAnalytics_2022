# In terminal:
    # cd cds-visual/portfolio/assignment_1/src
    # python compare_images.py
        
    
# parser
import argparse

# path tools
import sys, os
sys.path.append(os.path.join(".."))

# image processing
import cv2


# calculate distance between color histograms
def main():
    # define filepaths
    filepath1 = os.path.join("../input/image_0001.jpg")
    filepath2 = os.path.join("../input/image_0037.jpg")
    filepath3 = os.path.join("../input/image_0016.jpg")
    
    # load images
    image1 = cv2.imread(filepath1)
    image2 = cv2.imread(filepath2)
    image3 = cv2.imread(filepath3)
    
    # get histograms
    hist1 = cv2.calcHist([image1],
                         [0,1,2], 
                         None, 
                         [8,8,8], 
                         [0,256, 0,256, 0,256])
    
    hist2 = cv2.calcHist([image2], 
                         [0,1,2], 
                         None, 
                         [8,8,8], 
                         [0,256, 0,256, 0,256])
    
    hist3 = cv2.calcHist([image3], 
                         [0,1,2], 
                         None, 
                         [8,8,8], 
                         [0,256, 0,256, 0,256])
    
    # normalize histograms
    hist1_norm = cv2.normalize(hist1, hist1, 0,255, cv2.NORM_MINMAX)
    hist2_norm = cv2.normalize(hist2, hist2, 0,255, cv2.NORM_MINMAX) 
    hist3_norm = cv2.normalize(hist3, hist3, 0,255, cv2.NORM_MINMAX) 
    
    # get distance score between image1 and image2
    score1 = cv2.compareHist(hist1_norm,hist2_norm,cv2.HISTCMP_CHISQR)
    
    # get distance score between image1 and image3
    score2 = cv2.compareHist(hist1_norm,hist3_norm,cv2.HISTCMP_CHISQR)
    
    # get distance score between image2 and image3
    score3 = cv2.compareHist(hist2_norm,hist3_norm,cv2.HISTCMP_CHISQR)
    
    # print the three scores
    print(f"The distance between image1 and image2 is {score1}")
    print(f"The distance between image1 and image3 is {score2}")
    print(f"The distance between image2 and image3 is {score3}")  
        
        
# python program to execute
if __name__ == "__main__":
    main()
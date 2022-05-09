# In terminal:
    # cd cds-visual/portfolio/assignment_2/src
    # python logistic_regression.py

# path tools
import sys,os
sys.path.append(os.path.join(".."))

# image processing
import cv2

# numpy
import numpy as np

# cifar10 dataset from tensorflow
from tensorflow.keras.datasets import cifar10 

# tools from sklearn
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main():
    # load the CIFAR_10 data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # define labels
    labels = ["airplane",
             "automobile",
             "bird",
             "cat",
             "deer",
             "dog",
             "frog",
             "horse",
             "ship",
             "truck"]
    
    # convert data to grayscale and numpy array
    X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
    X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])

    # normalize the values
    X_train_scaled = X_train_grey/255
    X_test_scaled = X_test_grey/255

    # reshaping the X_train data
    nsamples, nx, ny = X_train_scaled.shape
    X_train_dataset = X_train_scaled.reshape((nsamples,nx*ny))
    
    # reshaping the X_test data
    nsamples, nx, ny = X_test_scaled.shape
    X_test_dataset = X_test_scaled.reshape((nsamples,nx*ny))

    # train a Logistic Regression classifier using scikit-learn
    clf = LogisticRegression(penalty = 'none',
                             tol = 0.1,
                             solver = 'saga',
                             multi_class = "multinomial").fit(X_train_dataset, y_train)
    # predict
    y_pred = clf.predict(X_test_dataset)
    
    # print the classification report to the terminal
    report = classification_report(y_test, y_pred, target_names = labels)
    print(report)

    # save the classification report 
    with open('output/lr_report.txt', 'a') as my_txt_file:
        my_txt_file.write(report)
        

# python program to execute
if __name__ == "__main__":
    main()
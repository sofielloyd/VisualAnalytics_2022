# In terminal:
# cd cds-visual/portfolio/assignment_3
# python src/transfer_learning.py


# operating system
import os 

# tf tools
import tensorflow as tf

# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
# cifar10 data - 32x32
from tensorflow.keras.datasets import cifar10

# generic model object
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential

# layers
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense,
                                     Dropout, 
                                     BatchNormalization)

# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

#scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# for plotting
import numpy as np
import matplotlib.pyplot as plt


# main function
def data():
    # # Load the CIFAR_10 data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # initialize label names for CIFAR-10 dataset
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
    
    # normalize
    X_train_norm = X_train/255
    X_test_norm = X_test/255
    
    # create one-hot encodings
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    
    return (X_train, y_train), (X_test, y_test), labels

def create_model():
    # load without classifier layer
    model = VGG16(include_top = False,
                  pooling = "avg",
                  input_shape = (32,32,3))
    
    # disable training of Conv layers
    for layer in model.layers:
        layer.trainable = False 
    
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation = 'relu')(flat1)
    output = Dense(10, activation = 'softmax')(class1)

    # define new model
    model = Model(inputs=model.inputs,
                  outputs=output)
    
    # compile
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 0.01,
    decay_steps = 10000,
    decay_rate = 0.9)
    sgd = SGD(learning_rate = lr_schedule)
    model.compile(optimizer = sgd,
                  loss = "categorical_crossentropy",
                  metrics = ["accuracy"])
    
    return model
    
    
def evaluate(model, X_test, y_test, labels):
    # evaluate
    predictions = model.predict(X_test, batch_size = 128)
    
    # Print the classification report to the terminal
    report = classification_report(y_test.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=labels)
    print(report)
    
    # Save the classification report to output/report.txt
    with open('output/report.txt', 'a') as my_txt_file:
        my_txt_file.write(report)
        

# plot history function
def plot_history(H, epochs):
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    
    # save model
    plt.savefig(os.path.join("output", "model.png"))
    
    
def main():
    (X_train, y_train), (X_test, y_test), labels = data()
    model = create_model()
    # train model
    H = model.fit(X_train, y_train,
                  validation_data = (X_test, y_test),
                  batch_size = 128,
                  epochs = 10,
                  verbose = 1)
    evaluate(model, X_test, y_test, labels)
    plot_history(H, 10)
 
        
# python program to execute
if __name__ == "__main__":
    main()
    

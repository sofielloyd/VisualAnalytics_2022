# operating system
import os

# for plotting
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

# tensorflow tools
import tensorflow as tf
import keras
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam,RMSprop,SGD,Adamax
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import (Conv2D,
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense,
                                     Input)

# confusion matrix and classification report
from sklearn.metrics import (confusion_matrix,
                             classification_report)


# get data
def data():
    # file path
    data_dir = '../input/dataset' 
   
    # training datasset
    train_ds = tf.keras.utils.image_dataset_from_directory(data_dir, 
                                                           validation_split=0.2,
                                                           subset="training",
                                                           color_mode='rgb',
                                                           seed=123,
                                                           image_size=(224, 224),
                                                           batch_size=32)

    # validation dataset
    val_ds  = tf.keras.utils.image_dataset_from_directory(data_dir,
                                                          validation_split=0.2,
                                                          subset="validation",
                                                          color_mode='rgb',
                                                          seed=123,
                                                          image_size=(224, 224),
                                                          batch_size=32)
    
    return train_ds, val_ds


# plot history
def plot_history(H, epochs):
    plt.style.use("seaborn-colorblind")

    # create canvas
    plt.figure(figsize=(12,6))
    # create subplot, 1 row, 2 colomns and use the first image
    plt.subplot(1,2,1)
    # take training loss from history and plot with each depth from 0 to 10
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    # and take validation loss, validation plot is going to be visualised as pots (linestyle)
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # tight layout
    plt.tight_layout()
    # adding legend
    plt.legend()

    # 1 row, 2 columns, use the second image 
    plt.subplot(1,2,2)
    # accuracy instead of loss
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    # save model
    plt.savefig(os.path.join('../output/trained_model.png'))


def create_model():
    # define model
    model = Sequential()
    
    # first set of layers 
    model.add(Conv2D(64, (3,3),   
                     padding = "same",
                     input_shape = (224,224,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2,2),              
                           strides = (2,2,)))

    # second set of layers 
    model.add(Conv2D(128, (5,5),
                     padding = "same")) 
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2,2),
                           strides = (2,2)))

    # third set of layers
    model.add(Conv2D(128, (5,5),
                     padding = "same")) 
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2,2),
                           strides = (2,2)))

    # FC => RELU
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))

    # softmax classifier
    model.add(Dense(5))
    model.add(Activation("softmax"))
    
    model.summary()
    
    # plot and save model 
    plot_model(model, to_file='../output/sequential_model.png', show_shapes = True, show_layer_names = True)
    
    # define optimizer
    optimizer = Adam(learning_rate=0.001)

    # compile model
    model.compile(loss = "sparse_categorical_crossentropy",
                  optimizer = optimizer,
                  metrics = ["accuracy"])
    
    return model
    
    
# evaluate model    
def evaluate_model(model, val_ds):
    test_label = np.concatenate([y for x, y in val_ds], axis=0)
    predictions = model.predict(val_ds)
    predictions.argmax(axis=1)
    
    # define labels
    labels = val_ds.class_names
    
    # get report
    report = classification_report(test_label, 
                                   predictions.argmax(axis=1),
                                   target_names = labels)
    # print report
    print(report)

    # save the classification report 
    with open('../output/report.txt', 'a') as my_txt_file:
        my_txt_file.write(report)
        
    return labels
    
    
def main(): 
    # clear models and parameters stored in memory
    tf.keras.backend.clear_session()
    # set parameters
    train_ds, val_ds = data()
    model = create_model()
    
    # train model
    history = model.fit(train_ds,
                        validation_data = val_ds,
                        batch_size=32,
                        epochs=20)
    
    # plot history
    plot_history(history, 20)
    
    # evaluate model
    evaluate_model(model, val_ds)
    
    # final evaluation 
    train_loss, train_acc = model.evaluate(train_ds)
    test_loss, test_acc = model.evaluate(val_ds)
    print("final train accuracy = {:.2f} , validation accuracy = {:.2f}".format(train_acc*100, test_acc*100))
    
    
# python program to execute
if __name__ == "__main__":
    main()
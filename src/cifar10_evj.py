## Session 7 task 2 - use tensorflow tools for the Cifar10 data (turn session7_cifar10_evj nb intp .py)

# path tools
import os
import cv2
# tools from sklearn
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
# tools from tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
# data loader
import numpy as np
from tensorflow.keras.datasets import cifar10
# matplotlib
import matplotlib.pyplot as plt

def get_data():
    #fetch the data and get the data and the labels
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # set labels to correct names
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
    # turn images into greyscale using list comprehensions. 
    X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
    X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])
    # Scaling images
    X_train_scaled = (X_train_grey)/255.0
    X_test_scaled = (X_test_grey)/255.0
    # reshaping images in training data
    nsamples, nx, ny = X_train_scaled.shape
    X_train_dataset = X_train_scaled.reshape((nsamples,nx*ny)) 
    #reshaping images in test data
    nsamples, nx, ny = X_test_scaled.shape
    X_test_dataset = X_test_scaled.reshape((nsamples,nx*ny))
    # label binarizing = convert labels to one-hot encoding
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)

    return X_train_dataset, y_train, X_test_dataset, y_test, lb

def nn_model(training_data, training_labels):
    # Creating model
    model = Sequential()
    model.add(Dense(256, 
                    input_shape=(1024,), #the input is the total number of pixel (32x32) in each image
                    activation="relu"))
    model.add(Dense(128, 
                    activation="relu"))
    model.add(Dense(10, 
                    activation="softmax"))  
    # overview of model
    print(model.summary())

    # train model using SGD
    sgd = SGD(0.01)  
    model.compile(loss="categorical_crossentropy",
                optimizer=sgd, 
                metrics=["accuracy"]) 

    history = model.fit(training_data, training_labels, 
                        epochs=10, 
                        batch_size=32)
    return model, history

def main():
    # load and prepare data
    X_train_dataset, y_train, X_test_dataset, y_test, lb = get_data()
    print("Data loaded and prepared")
    #create and train model
    model, history = nn_model(X_train_dataset, y_train)
    print("Model trained")
    # evaluate network
    print("[INFO] evaluating network...")
    predictions = model.predict(X_test_dataset, batch_size=32)
    # print classification report
    print(classification_report(y_test.argmax(axis=1),
                                predictions.argmax(axis=1), 
                                target_names=[str(x) for x in lb.classes_])) 
    
if __name__ == "__main__":
    main()
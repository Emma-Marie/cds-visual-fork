## Turn notebook 7 into .py script

# generic tools
import numpy as np
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

def get_data():
    #fetch the data and get the data and the labels
    data, labels = fetch_openml('mnist_784', version=1, return_X_y=True) 
    # normalise data
    data = data.astype("float")/255.0
    # split data
    (X_train, X_test, y_train, y_test) = train_test_split(data,
                                                        labels, 
                                                        test_size=0.2) 
    # Convert labels to one-hot encoding
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    print("Data prepared")
    
    return X_train, y_train, X_test, y_test, lb

def nn_model(X_train, y_train, X_test):
    # Creating the model, define architecture 784x256x128x10
    model = Sequential() 
    model.add(Dense(256, 
                    input_shape=(784,), 
                    activation="relu")) 
    model.add(Dense(128, 
                    activation="relu"))
    model.add(Dense(10, 
                    activation="softmax")) 
    print("Model created")
    # Overview of the model
    model.summary()
    # train model using SGD
    sgd = SGD(0.01) 
    model.compile(loss="categorical_crossentropy",
                optimizer=sgd, 
                metrics=["accuracy"])

    history = model.fit(X_train, y_train, 
                        epochs=10, 
                        batch_size=32)
    # Evaluate model
    print("[INFO] evaluating network...")
    predictions = model.predict(X_test, batch_size=32)
    print("Model trained")
    
    return history, model, predictions

def main():
    # get and prepare data
    X_train, y_train, X_test, y_test, lb = get_data()
    # create and train model
    history, model, predictions = nn_model(X_train, y_train, X_test)
    # Classification report
    print(classification_report(y_test.argmax(axis=1),
                                predictions.argmax(axis=1), 
                                target_names=[str(x) for x in lb.classes_]))
    
if __name__ == "__main__":
    main()
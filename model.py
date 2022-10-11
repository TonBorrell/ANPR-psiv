from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import joblib
from sklearn import svm
import numpy as np
import os
import cv2
from data_augmentation import *


def knn(data, labels, digit):

    # Split data into test and train data
    nsamples, nx, ny = data.shape
    dataset = data.reshape((nsamples, nx * ny))
    x_train, x_test, y_train, y_test = train_test_split(
        dataset, labels, test_size=0.33, random_state=42
    )

    # KNN class instance
    clf = KNeighborsClassifier(n_neighbors=3)

    # Training
    clf.fit(x_train, y_train)

    # Testing
    y_pred = clf.predict(x_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Accuracy
    accuracy = clf.score(x_test, y_test)

    # Saving the model
    joblib.dump(clf, f"model/knn/model_{digit}")

    print("\n-- K Nearest Neighbors --")
    print("Training completed")
    print("Accuracy : " + str(accuracy))
    print("Confusion Matrix :")
    print(cm)

    return clf


def mlp(data, labels, digit):

    # Split data into test and train data
    nsamples, nx, ny = data.shape
    dataset = data.reshape((nsamples, nx * ny))
    x_train, x_test, y_train, y_test = train_test_split(
        dataset, labels, test_size=0.33, random_state=42
    )

    # Multi layer perceptron class instance
    clf = MLPClassifier(
        solver="lbfgs",
        alpha=1e-4,
        hidden_layer_sizes=200,
        random_state=1,
    )

    # Training
    clf.fit(x_train, y_train)

    # Testing
    y_pred = clf.predict(x_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Accuracy
    accuracy = clf.score(x_test, y_test)

    # Saving the model
    joblib.dump(clf, f"model/mlp/model_{digit}")

    print("\n-- Multi Layer Perceptron --")
    print("Training completed")
    print("Accuracy : " + str(accuracy))
    print("Confusion Matrix :")
    print(cm)


def svm_function(data, labels, digit):

    # Split data into test and train data
    nsamples, nx, ny = data.shape
    dataset = data.reshape((nsamples, nx * ny))
    x_train, x_test, y_train, y_test = train_test_split(
        dataset, labels, test_size=0.33, random_state=42
    )

    # SVM class instance
    clf = svm.SVC(kernel="linear", C=1, gamma=1)

    # Training
    clf.fit(x_train, y_train)

    # Testing
    y_pred = clf.predict(x_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Accuracy
    accuracy = clf.score(x_test, y_test)

    # Saving the model
    joblib.dump(clf, f"model/svm/model_{digit}")

    print("\n-- SVM --")
    print("Training completed")
    print("Training Accuracy : " + str(accuracy))
    print("Confusion Matrix :")
    print(cm)


def get_images_numbers():
    images = os.listdir("dataset/real/resized/")
    labels = []
    images_read = []
    augmentation = [image_rotation, image_crop, image_dilate, image_translation]
    for i in images:
        label = i.split("_", 1)[1]
        label = label.split("+", 1)[0]
        image = cv2.imread("dataset/real/resized/" + i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for j in augmentation:
            for i in range(25):
                images_read.append(j(image))
                labels.append(label)
        images_read.append(image)
        labels.append(label)

    model_knn = knn(np.array(images_read), labels, "numbers")
    model_mlp = mlp(np.array(images_read), labels, "numbers")
    model_svm = svm_function(np.array(images_read), labels, "numbers")

    return model_knn, model_mlp, model_svm


def get_images_letters():
    images = os.listdir("dataset_letters/all/")
    labels = []
    images_read = []
    augmentation = [image_rotation, image_crop, image_dilate, image_translation]
    for i in images:
        label = i.split("_", 1)[1]
        label = label.split(".", 1)[0]
        label = label.split("+", 1)[0]
        image = cv2.imread("dataset_letters/all/" + i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for j in augmentation:
            for i in range(25):
                images_read.append(j(image))
                labels.append(label)
        images_read.append(image)
        labels.append(label)

    model_knn = knn(np.array(images_read), labels, "letters")
    model_mlp = mlp(np.array(images_read), labels, "letters")
    model_svm = svm_function(np.array(images_read), labels, "letters")

    return model_knn, model_mlp, model_svm


get_images_numbers()
# get_images_letters()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# for metrics and parallel processing
import time
import joblib
import random
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED

from utils import mnist_reader 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import  LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

FashionLabel = {
  0 : "T-shirt/top",
  1 : "Trouser",
  2 : "Pullover",
  3 : "Dress",
  4 : "Coat",
  5 : "Sandal",
  6 : "Shirt",
  7 : "Sneaker",
  8 : "Bag",
  9 : "Ankle boot",
}

def viewMatAsImg(data):
  """ Function to draw the 28 x 28 flattened image as a 28x28 grayscale image """
  preview = data 
  preview = preview.reshape((28, 28))
  plt.imshow(preview, 'gray')
  plt.show()

# Increases the contrast of an image
# Factor can be changed to yield better results
def increaseContrast(img, factor=20):
  img = img.reshape(28,28)
  factor = float(factor)
  res = np.clip(128 + factor * img - factor * 128, 0, 255).astype(np.uint8)
  return res.flatten()

def normalize(img):
  res = img / 255
  return res

# generic function to train a model 
@ignore_warnings(category=ConvergenceWarning)
def train(model, X, Y, mode, start_time):
  res = model.fit(X, Y)
  elapsed = time.time() - start_time
  print(f"{mode} has done. Time elasped {elapsed} seconds")
  return model, mode

# generic function to run cross validation 
@ignore_warnings(category=ConvergenceWarning)
def crossValidate(model, X, Y, cv, mode, start_time):
  res = cross_val_predict(model, X, Y, cv=cv)
  elapsed = time.time() - start_time
  print(f"{mode} has done. Time elapsed: {elapsed} seconds")
  return model, res, mode

# only train when it is run as main program
if __name__ == "__main__":
  X_pre, Y_pre= mnist_reader.load_mnist('data', kind="train")
  # Bundle 2 array together and run a shuffle to it
  bundled = []
  for i in range(len(X_pre)):
    bundled.append([X_pre[i], Y_pre[i]])
  random.shuffle(bundled)

  # de bundled the dataset
  X = np.array([ x[0] for x in bundled])
  Y = np.array([ y[1] for y in bundled])

  # preprocess all data 
  X = np.array(list(map(increaseContrast,X)))
  X = np.array(list(map(normalize,X)))

  # 0.7 for training, 0.2 for testing, 0.1 for cross validation
  t_point = int(round(0.7 * len(X)))
  latter_bound = int(round(0.2 * len(X)))

  # Splitting dataset 
  X_train, Y_train = X[:t_point], Y[:t_point]
  X_test, Y_test = X[t_point: t_point+latter_bound], Y[t_point :t_point + latter_bound]
  cv_X, cv_Y = X[t_point+latter_bound:], Y[t_point+latter_bound:]

  print(f"Training set X, Y Dimensions : {X_train.shape} {Y_train.shape}")
  print(f"Testing set X, Y Dimensions : {X_test.shape} {Y_test.shape}")
  print(f"Cross validation set X, Y Dimensions : {cv_X.shape} {cv_Y.shape}")

  svc = None; rfc = None; lrgs = None
  svc_scores = None; rfc_scores = None; lrgs_scores = None

  print("##### Start Training #####")
  #SVC
  svc = SVC(gamma='scale')
  start_time = time.time()
  svc, _ = train(svc, X_train, Y_train, "SVC", start_time)
  # RFC
  start_time = time.time()
  rfc = RandomForestClassifier(n_estimators=100)
  rfc, _ = train(rfc, X_train, Y_train, "RFC", start_time)
  # LGRS
  start_time = time.time()
  lrgs = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=5000)
  lrgs, _ = train(lrgs, X_train, Y_train, "LRGS", start_time)
  print("#############\n")

  # print("##### Saving Classifiers #####\n")
  # joblib.dump(svc, "saved_model/Classifier-svc")
  # joblib.dump(rfc, "saved_model/Classifier-rfc")
  # joblib.dump(lrgs, "saved_model/Classifier-lrgs")

  print("##### Start Cross Validation #####")
  start_time = time.time()
  _, svc_scores, _ = crossValidate(svc, cv_X, cv_Y, 10, "SVC", start_time)
  start_time = time.time()
  _, rfc_scores, _ = crossValidate(rfc, cv_X, cv_Y, 10, "RFC", start_time)
  start_time = time.time()
  _, lrgs_scores, _ = crossValidate(lrgs, cv_X, cv_Y, 10, "LRGS", start_time)
  print("#############\n")

  print("##### Cross Validation Accuracy #####")
  print(f"Support Vector Classifier: {accuracy_score(cv_Y,svc_scores)}")
  print(f"Random Forest Classifier: {accuracy_score(cv_Y,rfc_scores)}")
  print(f"Logistic Regression Classifier: {accuracy_score(cv_Y,lrgs_scores)}")
  print("#############\n")
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# for metrics and parallel processing
import time
import joblib
import random

from utils import mnist_reader 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import  LogisticRegression
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

# Increases the contrast of an image
# Factor can be changed to yield better results
def increaseContrast(img, factor=20):
  img = img.reshape(28,28)
  factor = float(factor)
  res = np.clip(128 + factor * img - factor * 128, 0, 255).astype(np.uint8)
  return res.flatten()

# Scale down the pixels' values on a scale of 255
def normalize(img):
  res = img / 255
  return res

# generic function to predict using a model
def predict(model, X, mode, start_time):
  res = model.predict(X)
  elapsed = time.time() - start_time 
  print(f"{mode} has done. Time elapsed : {elapsed} seconds")
  return model, res, mode

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
  res = cross_val_score(model, X, Y, cv=cv)
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
  Y_train = np.array([ y[1] for y in bundled])

  # preprocess all data 
  # X = np.array(list(map(increaseContrast,X)))
  X_train = np.array(list(map(normalize,X)))

  # 0.9 for training, 0.1 for cross validation 
  # t_point = int(round(1 * len(X)))

  # Splitting dataset 
  # X_train, Y_train = X[:t_point], Y[:t_point]
  # X_test, Y_test = X[t_point:], Y[t_point :]
  # cv_X, cv_Y = X[t_point+latter_bound:], Y[t_point+latter_bound:]

  print(f"Training set X, Y Dimensions : {X_train.shape} {Y_train.shape}")
  # print(f"Cross Validation set X, Y Dimensions : {X_test.shape} {Y_test.shape}")

  svc = None; rfc = None; lrgs = None
  svc_scores = None; rfc_scores = None; lrgs_scores = None

  svc = SVC(gamma='scale', decision_function_shape="ovr")
  rfc = RandomForestClassifier(n_estimators=500, criterion="entropy", n_jobs=-1)
  lrgs = LogisticRegression(solver='lbfgs', multi_class='ovr', max_iter=5000)

  print("##### Start Cross Validation #####")
  start_time = time.time()
  _, svc_scores, _ = crossValidate(svc, X_train, Y_train, 6, "SVC", start_time)
  start_time = time.time()
  _, rfc_scores, _ = crossValidate(rfc, X_train, Y_train, 6, "RFC", start_time)
  start_time = time.time()
  _, lrgs_scores, _ = crossValidate(lrgs, X_train, Y_train, 6, "LRGS", start_time)
  print("#############\n")

  print("##### Cross Validation Accuracy #####")
  print("Support Vector Classifier: %.4f (+/- %.4f)" % (svc_scores.mean(), svc_scores.std()*2))
  print("Random Forest Classifier: %.4f (+/- %.4f)" % (rfc_scores.mean(), rfc_scores.std()*2))
  print("Logistic Regression Classifier: %.4f (+/- %.4f)" % (lrgs_scores.mean(), lrgs_scores.std()*2))
  print("#############\n")

  print("##### Start Training #####")
  #SVC
  start_time = time.time()
  svc, _ = train(svc, X_train, Y_train, "SVC", start_time)
  # RFC
  start_time = time.time()
  rfc, _ = train(rfc, X_train, Y_train, "RFC", start_time)
  # LGRS
  start_time = time.time()
  lrgs, _ = train(lrgs, X_train, Y_train, "LRGS", start_time)
  print("#############\n")

  # Uncomment when all parameters are found
  print("##### Saving Classifiers #####\n")
  joblib.dump(svc, "saved_model/Classifier-svc")
  joblib.dump(rfc, "saved_model/Classifier-rfc")
  joblib.dump(lrgs, "saved_model/Classifier-lrgs")

  # print("##### Test on Validation set #####")
  # start_time = time.time()
  # svc, svc_predict, _ = predict(svc, X_test, "SVC", start_time)
  # start_time = time.time()
  # rfc, rfc_predict, _ = predict(rfc, X_test, "RFC", start_time)
  # start_time = time.time()
  # lrgs, lrgs_predict, _ = predict(lrgs, X_test, "LRGS", start_time)
  # print("#############\n")

  # print("##### Accuracy on Validation set #####")
  # print(f"Total test data : {len(Y_test)}")
  # print(f"Support Vector Machine : {accuracy_score(Y_test, svc_predict)}")
  # print(f"Random Forest : {accuracy_score(Y_test, rfc_predict)}")
  # print(f"Logistic Regression : {accuracy_score(Y_test, lrgs_predict)}")
  # print("#############\n")
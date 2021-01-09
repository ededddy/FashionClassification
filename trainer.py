#!/usr/bin/env python
# -*- coding: utf-8 -*-

# for metrics and parallel processing
import time
import joblib
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

# generic function to train a model used for parallel training
@ignore_warnings(category=ConvergenceWarning)
def train(model, X, Y, mode, start_time):
  res = model.fit(X, Y)
  elapsed = time.time() - start_time
  print(f"{mode} has done. Time elasped {elapsed} seconds")
  return model, mode

@ignore_warnings(category=ConvergenceWarning)
def crossValidate(model, X, Y, cv, mode, start_time):
  res = cross_val_predict(model, X, Y, cv=cv)
  elapsed = time.time() - start_time
  print(f"{mode} has done. Time elapsed: {elapsed} seconds")
  return model, res, mode

# only train when it is run as main program
if __name__ == "__main__":
  X, Y= mnist_reader.load_mnist('data', kind="train")
  # 0.7 for training, 0.2 for testing, 0.1 for cross validation
  t_point = int(round(0.7 * len(X)))
  latter_bound = int(round(0.2 * len(X)))

  X_train, Y_train = X[:t_point], Y[:t_point]
  X_test, Y_test = X[t_point: t_point+latter_bound], Y[t_point :t_point + latter_bound]
  cv_X, cv_Y = X[t_point+latter_bound:], Y[t_point+latter_bound:]
  # preprocess all data 
  X_train = np.array(list(map(increaseContrast, X_train)))
  X_test = np.array(list(map(increaseContrast, X_test)))
  cv_X = np.array(list(map(increaseContrast, cv_X)))
  print(f"Training set X, Y Dimenstions : {X_train.shape} {Y_train.shape}")
  print(f"Testing set X, Y Dimenstions : {X_test.shape} {Y_test.shape}")
  print(f"Cross validation set X, Y Dimenstions : {cv_X.shape} {cv_Y.shape}")
  svc = None; rfc = None; lgrs = None
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
  lgrs = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=5000)
  lgrs, _ = train(lgrs, X_train, Y_train, "LGRS", start_time)
  print("#############\n")
  # print("##### Saving Classifiers #####")
  # joblib.dump(svc, "saved_model/Classifier-svc")
  # joblib.dump(rfc, "saved_model/Classifier-rfc")
  # joblib.dump(lrgs, "saved_model/Classifier-lgrs")
  print("##### Start Cross Validation #####")
  start_time = time.time()
  _, svc_scores, _ = crossValidate(svc, cv_X, cv_Y, 10, "SVC", start_time)
  start_time = time.time()
  _, rfc_scores, _ = crossValidate(rfc, cv_X, cv_Y, 10, "RFC", start_time)
  start_time = time.time()
  _, lgrs_scores, _ = crossValidate(lgrs, cv_X, cv_Y, 10, "LRGS", start_time)
  print("#############\n")
  print("##### Cross Validation Accuracy #####")
  print(f"Support Vector Classifier: {accuracy_score(cv_Y,svc_scores)}")
  print(f"Random Forest Classifier: {accuracy_score(cv_Y,rfc_scores)}")
  print(f"Logistic Regression Classifier: {accuracy_score(cv_Y,lrgs_scores)}")


# The code block below is for parallel training, however my system does not have enough ram for it
  # print("##### Training #####")
  # with ProcessPoolExecutor() as executor :
  #   futures = []
  #   start_time = time.time()
  #   # Train SVC
  #   svc = SVC(gamma='scale')
  #   futures.append(
  #     executor.submit(
  #       train, svc, X_train, Y_train, "SVC", start_time
  #     )
  #   )
  #   # Train RandomForest Classifier
  #   rfc = RandomForestClassifier(n_estimators=100)
  #   futures.append(
  #     executor.submit(
  #       train, rfc, X_train, Y_train, "RFC", start_time
  #     )
  #   )
  #   # Train LogisticRegression Classifier
  #   lrgs = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=5000)
  #   futures.append(
  #     executor.submit(
  #       train, lrgs, X_train, Y_train, "LRGS", start_time
  #     )
  #   )
  #   #wait until all parallel tasks are done
  #   wait(futures, return_when=ALL_COMPLETED)
  #   for future in futures:
  #     model, mode = future.result()
  #     if mode == "SVC":
  #       svc = model
  #     elif mode == "RFC":
  #       rfc = model
  #     elif mode == "LRGS":
  #       lrgs = model
  # print("#################\n")

  # print("##### Saving Classifiers #####")
  # joblib.dump(svc, "saved_model/Classifier-svc")
  # joblib.dump(rfc, "saved_model/Classifier-rfc")
  # joblib.dump(lrgs, "saved_model/Classifier-lgrs")

  # print("##### Cross Validation #####")
  # with ProcessPoolExecutor() as executor :
  #   futures = []
  #   start_time = time.time()

  #   # SVC
  #   futures.append(
  #     executor.submit(
  #       crossValidate, svc, cv_X, cv_Y, 10, "SVC", start_time
  #     )
  #   )
  #   # RFC
  #   futures.append(
  #     executor.submit(
  #       crossValidate, rfc, cv_X, cv_Y, 10, "SVC", start_time
  #     )
  #   )
  #   # LRGS
  #   futures.append(
  #     executor.submit(
  #       crossValidate, lrgs, cv_X, cv_Y, 10, "SVC", start_time
  #     )
  #   )
  #   wait(futures, return_when=ALL_COMPLETED)
  #   for future in futures :
  #     model, cv_scores, mode = future.result()
  #     if mode == "SVC":
  #       svc_scores = cv_scores
  #     elif mode == "RFC":
  #       rfc_scores = cv_scores
  #     elif mode == "LRGS":
  #       lrgs_scores = cv_scores
  # print("#################\n")

  # print("##### Cross Validation Scoring #####")
  # print(f"Support Vector Classifier: {accuracy_score(cv_Y,svc_scores)}")
  # print(f"Random Forest Classifier: {accuracy_score(cv_Y,rfc_scores)}")
  # print(f"Logistic Regression Classifier: {accuracy_score(cv_Y,lrgs_scores)}")

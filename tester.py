import time
import numpy as np
import joblib
from prettytable import PrettyTable

from utils import mnist_reader
import matplotlib.pyplot as plt
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import  LogisticRegression
from sklearn.model_selection import cross_val_predict

# Increases the contrast of an image
# Factor can be changed to yield better results
def increaseContrast(img, factor=20):
  img = img.reshape(28,28)
  factor = float(factor)
  res = np.clip(128 + factor * img - factor * 128, 0, 255).astype(np.uint8)
  return res.flatten()

def predict(model, X, mode, start_time):
  res = model.predict(X)
  elapsed = time.time() - start_time 
  print(f"{mode} has done. Time elapsed : {elapsed} seconds")
  return model, res, mode

def readClassifier(url, mode):
  classifier = joblib.load(url)
  return classifier, mode

if __name__ == "__main__":
  X_test, Y_test = mnist_reader.load_mnist('data', kind="t10k")
  X_test = np.array(list(map(increaseContrast, X_test)))

  print("##### Read Classifiers #####")
  svc = readClassifier("saved_model/Classifier-svc", "SVC")
  rfc = readClassifier("saved_model/Classifier-rfc", "RFC")
  lgrs = readClassifier("saved_model/Classifier-lgrs", "lgrs")
  print("#############\n")

  start_time = time.time()
  print("##### Predicting #####")
  svc, svc_predict, _ = predict(svc, X_test, "SVC", start_time)
  rfc, rfc_predict, _ = predict(rfc, X_test, "SVC", start_time)
  lgrs, lgrs_predict, _ = predict(lgrs, X_test, "SVC", start_time)
  print("#############\n")

  # Calculate Accuracy
  test_total = len(Y_test)
  svc_correct_cnt = 0
  rfc_correct_cnt = 0
  lgrs_correct_cnt = 0
  for i in range(test_total):
    if (svc_predict[i] == Y_test[i]) :
      svc_correct_cnt += 1
    if (rfc_predict[i] == Y_test[i]) :
      rfc_correct_cnt += 1
    if (lgrs_predict[i] == Y_test[i]) :
      lgrs_correct_cnt += 1
  
  svc_accur = svc_correct_cnt / test_total
  rfc_accur = rfc_correct_cnt / test_total
  lgrs_accur = lgrs_correct_cnt / test_total
  
  print("##### Accuracy #####")
  print(f"Total test data : {test_total}")
  print(f"Support Vector Machine : {svc_accur}")
  print(f"Random Forest : {rfc_accur}")
  print(f"Logistic Regression : {lgrs_accur}")

  table = PrettyTable(['', 'SVC', 'RFC', 'LRC'])
  table.add_row(['Accuracy', svc_accur, rfc_accur, lgrs_accur])
  print(table)
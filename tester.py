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
import collections 


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
svcMissLabels = list(range(10))
rfcMissLabels = list(range(10))
lrgsMissLabels = list(range(10))
LabelsCnt = list(range(10))

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
  # Preprocess
  # X_test = np.array(list(map(increaseContrast, X_test)))
  X_test = np.array(list(map(normalize, X_test)))

  print("##### Read Classifiers #####\n")
  svc, _ = readClassifier("saved_model/Classifier-svc", "SVC")
  rfc, _ = readClassifier("saved_model/Classifier-rfc", "RFC")
  lrgs, _ = readClassifier("saved_model/Classifier-lrgs", "LRGS")

  start_time = time.time()
  print("##### Predicting #####")
  svc, svc_predict, _ = predict(svc, X_test, "SVC", start_time)
  rfc, rfc_predict, _ = predict(rfc, X_test, "RFC", start_time)
  lrgs, lrgs_predict, _ = predict(lrgs, X_test, "LRGS", start_time)
  print("#############\n")

  # Calculate Accuracy
  test_total = len(Y_test)
  svc_correct_cnt = 0
  rfc_correct_cnt = 0
  lrgs_correct_cnt = 0
  LabelsCnt = collections.Counter(Y_test)
  for i in range(test_total):
    if (svc_predict[i] == Y_test[i]) :
      svc_correct_cnt += 1
    else :
      svcMissLabels[Y_test[i]] += 1
    if (rfc_predict[i] == Y_test[i]) :
      rfc_correct_cnt += 1
    else :
      rfcMissLabels[Y_test[i]] += 1
    if (lrgs_predict[i] == Y_test[i]) :
      lrgs_correct_cnt += 1
    else :
      lrgsMissLabels[Y_test[i]] += 1
  
  svc_accur = svc_correct_cnt / test_total
  rfc_accur = rfc_correct_cnt / test_total
  lrgs_accur = lrgs_correct_cnt / test_total
  
  print("##### Accuracy #####")
  print(f"Total test data : {test_total}")
  print(f"Support Vector Machine : {svc_accur}")
  print(f"Random Forest : {rfc_accur}")
  print(f"Logistic Regression : {lrgs_accur}")
  print("#############\n")

  table = PrettyTable(['', 'SVC', 'RFC', 'LRC'])
  table.add_row(['Accuracy', svc_accur, rfc_accur, lrgs_accur])
  print(table)
  print()
  misslabel_table = PrettyTable([
    'ID', 'Label', 'SVC Error', 'RFC Errror', 'LRGS Error', 'Total'
  ])
  for (key, val) in FashionLabel.items() :
    misslabel_table.add_row([
      key, val, 
      "%.3f" % ( svcMissLabels[key]/LabelsCnt[key])  , 
      "%.3f" % ( rfcMissLabels[key]/LabelsCnt[key])  , 
      "%.3f" % ( lrgsMissLabels[key]/LabelsCnt[key]) ,
      LabelsCnt[key]  
    ])
  print(misslabel_table)
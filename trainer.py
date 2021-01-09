# for metrics and parallel processing
import time
import joblib
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
import random

from utils import mnist_reader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import  LogisticRegression
from sklearn.model_selection import cross_val_predict

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

def increaseContrast(img, factor=20):
  img = img.reshape(28,28)
  factor = float(factor)
  res = np.clip(128 + factor * img - factor * 128, 0, 255).astype(np.uint8)
  return res.flatten()

# generic function to train a model used for parallel training
def train(model, X, Y, mode, start_time):
  result = model.fit(X, Y)
  elapsed = time.time() - start_time
  print(f"{mode} has done. Time elasped {elapsed} seconds")
  return model, mode

def crossValidate(model, X, Y, cv, mode, start_time):
  res = cross_val_predict(model, X, Y, cv=cv)
  elapsed = time.time() - start_time
  print(f"{mode} has done. Time elapsed: {elapsed} seconds")
  return model, res, mode

# only train when it is run as main program
if __name__ == "__main__":
  X_train, Y_train = mnist_reader.load_mnist('data', kind="train")
  X_test, Y_test = mnist_reader.load_mnist('data', kind="t10k")
  X_train = np.array(list(map(increaseContrast, X_train)))
  X_test = np.array(list(map(increaseContrast, X_test)))
  viewMatAsImg(X_train[0])


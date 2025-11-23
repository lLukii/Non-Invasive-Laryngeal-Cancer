import numpy as np
import matplotlib.pyplot as plt
import librosa
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from cfg import *

def pad_seq(x, pad_len):
  if len(x) > pad_len: x = x[:pad_len]
  else: x = np.pad(x, (0, pad_len - len(x)))
  return x

sr = 16000
padding = 48000
dataset = 'ALL'

features, labels = [], []
for voice_type in ["Benign", "Malignant"]:
    for sample in os.listdir(os.path.join(femh_path, voice_type)):
        y, sr = librosa.load(os.path.join(femh_path, voice_type, sample), sr=16000)
        y = pad_seq(y, padding)
        y = librosa.feature.mfcc(y=y, sr=sr)
        features.append(y)
        labels.append(voice_type == "Malignant")

print("FEMH Dataset Complete")
malignant_cond = ["Stimmlippenkarzinom", "Hypopharynxtumor", "Kehlkopftumor",
                    "Epiglottiskarzinom", "Mesopharynxtumor", "Carcinoma in situ",
  "Dysplastische Dysphonie", "Dysplastischer Kehlkopf"]
for condition in os.listdir(svd_path):
  for sample in os.listdir(os.path.join(svd_path, condition)):
    path = os.path.join(svd_path, condition, sample)
    if path.endswith(".wav"):
      x, sr = librosa.load(path, sr=16000)
      x = pad_seq(x, padding)
      x = librosa.feature.mfcc(y=x, sr=sr)
      features.append(x)
      labels.append(condition in malignant_cond)

features = np.array(features)
labels = np.array(labels)
scalar = StandardScaler()
features = scalar.fit_transform(features)

train_x, val_x, train_y, val_y = train_test_split(features, labels, test_size=0.2, random_state=42)
param_grid = {"loss" : ['hinge', 'log_loss'],
              "max_iter" : [500, 1000, 2000],
              "tol" : [1e-5, 1e-4, 1e-3]}

train_acc, val_acc = [], []
conf_matricies, reports = [], []
for num_experiments in range(10):
  grid_search = GridSearchCV(
      SGDClassifier(max_iter=1000, tol=1e-3, class_weight="balanced"),
      param_grid, cv=5, verbose=1, scoring='balanced_accuracy')

  grid_search.fit(train_x, train_y)
  clf = grid_search.best_estimator_
  train_pred = clf.predict(train_x)
  val_pred = clf.predict(val_x)

  train_acc.append(balanced_accuracy_score(train_y, train_pred))
  val_acc.append(balanced_accuracy_score(val_y, val_pred))
  conf_matricies.append(confusion_matrix(train_y, train_pred))
  reports.append(classification_report(train_y, train_pred))

plt.plot(train_acc)
plt.plot(val_acc)
plt.title("Grid Search - Best Model vs Accuracy")
plt.legend(["Train", "Validation"])
plt.xlabel("Experiment")
plt.ylabel("Balanced Accuracy")
plt.show()

print("Train Average ", sum(train_acc) / len(train_acc))
print("Val Average ", sum(val_acc) / len(val_acc))
print("Train Std ", np.std(train_acc))
print("Val Std ", np.std(val_acc))
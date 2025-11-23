from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import seaborn as sns
import librosa
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from cfg import *
from src.dataset import AudioData, StratifiedSampler
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

import torch.nn as nn
from models import Wav2VecClassifier, ContrastiveLoss

def assess_model_perf(model, data, show_cm=False):
  model.eval()
  y_pred = []
  y_true = []
  with torch.no_grad():
      for inputs, y in data:
          x, a = inputs
          _, pred = model(x.to(device), a.to(device))
          pred = pred.argmax(1).cpu().numpy()
          y_pred.extend(pred)
          y_true.extend(y.cpu().numpy())

  metrics = classification_report(y_true, y_pred)
  cm = confusion_matrix(y_true, y_pred)
  if show_cm: sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

  return metrics, balanced_accuracy_score(y_true, y_pred)

scaler = MinMaxScaler(feature_range=(-1, 1))
femh_path = "/content/drive/MyDrive/ISEF/Datasets/FEMH"
signals, labels = [], []
for voice_type in ["Benign", "Malignant"]:
    for sample in os.listdir(os.path.join(femh_path, voice_type)):
        y, sr = librosa.load(os.path.join(femh_path, voice_type, sample), sr=16000)
        signals.append(scaler.fit_transform(y.reshape(-1, 1)).squeeze())
        labels.append(voice_type == "Malignant")

print("FEMH Dataset complete")
malignant_cond = ["Stimmlippenkarzinom", "Hypopharynxtumor", "Kehlkopftumor",
                    "Epiglottiskarzinom", "Mesopharynxtumor", "Carcinoma in situ",
  "Dysplastische Dysphonie", "Dysplastischer Kehlkopf"]
svd_path = "/content/drive/MyDrive/ISEF/Datasets/SVD"
for condition in os.listdir(svd_path):
  for sample in os.listdir(os.path.join(svd_path, condition)):
    path = os.path.join(svd_path, condition, sample)
    if path.endswith(".wav"):
      x, sr = librosa.load(path, sr=16000)
      signals.append(x)
      labels.append(condition in malignant_cond)

labels = torch.tensor(labels, dtype=torch.long)

train_x, val_x, train_y, val_y = train_test_split(signals, labels, test_size=0.2)
train_dataset = AudioData(train_x, train_y, augment=True)
val_dataset = AudioData(val_x, val_y)

sampler = StratifiedSampler(train_y, batch_size, 1)
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([1., batch_size]))
contrastive_loss = ContrastiveLoss().to(device)
model = Wav2VecClassifier().to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr)
lr_scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=2)

train_loss, val_loss = [], []
def test_accuracy(data):
    model.eval()
    acc, loss_total = 0, 0
    c_loss = 0
    size = len(data.dataset)
    num_batches = (size + batch_size - 1) // batch_size
    with tqdm(desc="Testing in progress....", total=num_batches) as pbar:
      with torch.no_grad():
        for inputs, y in data:
            x, a = inputs
            x, a, y = x.to(device), a.to(device), y.to(device)
            emb, logit = model(x, a)
            loss = ce_loss(logit, y)
            try: c_loss += c_weight * contrastive_loss(emb, y).item()
            except RuntimeError: pass
            loss_total += loss.item()
            acc += (logit.argmax(1) == y).sum().item()
            pbar.update(1)

    if data is val_loader:
        val_loss.append(loss_total / num_batches)
    else:
        train_loss.append(loss_total / num_batches)

    return loss_total / num_batches, c_loss / num_batches, acc / size

def train_epoch(data):
    model.train()
    size = len(data.dataset)
    with tqdm(desc="Training in progress....", total=(size + batch_size - 1) // batch_size) as pbar:
      for input1, y in data:
          x, a = input1
          x, a, y = x.to(device), a.to(device), y.to(device)
          emb, logit = model(x, a)
          # backward
          optim.zero_grad()
          loss = ce_loss(logit, y) + c_weight * contrastive_loss(emb, y)
          loss.backward()
          optim.step()
          pbar.update(1)

def run_training():
  for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}")
    train_epoch(train_loader)
    t_loss, c_loss, t_acc = test_accuracy(train_loader)
    loss, c_loss2, acc = test_accuracy(val_loader)
    lr_scheduler.step(loss)
    print("Training CE Loss, and Accuracy", t_loss, c_loss, t_acc)
    print("Validation CE Loss, and Accuracy", loss, c_loss2, acc)
    _, b_acc = assess_model_perf(model, val_loader)
    print("Balanced Acc: ", b_acc)



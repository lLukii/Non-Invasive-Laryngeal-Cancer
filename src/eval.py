from cfg import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.decomposition import PCA


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

def assess_ensemble_perf(models, data):
  y_pred = []
  y_true = []
  with torch.no_grad():
      for inputs, y in data:
          x, a = inputs
          x = x.to(device)
          a = a.to(device)
          preds = torch.zeros((num_models, x.shape[0]))
          for i, model in enumerate(models):
            _, logits = model(x, a)
            pred = logits.argmax(dim=1) # (B, num_models)
            preds[i] = pred
          pred = preds.mode(dim=0).values.squeeze()
          y_pred.extend(pred.cpu().numpy())
          y_true.extend(y.cpu().numpy())

  metrics = classification_report(y_true, y_pred)
  cm = confusion_matrix(y_true, y_pred)
  sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

  return metrics, balanced_accuracy_score(y_true, y_pred)

def visualize_embeddings(models, data):
    model = models[-1]
    positive = []
    negative = []
    with torch.no_grad():
        for data, y in data:
            x, a = data
            x = x.to(device)
            a = a.to(device)
            _, _, emb = model(x, a, True)
            positive.extend(emb[y == 1].cpu().numpy())
            negative.extend(emb[y == 0].cpu().numpy())

    reduction = PCA(n_components=2)
    positive = reduction.fit_transform(positive)
    negative = reduction.fit_transform(negative)

    plt.title("Visual representation of wav2vec embeddings")
    plt.scatter(positive[:, 0], positive[:, 1], label="Positive", color="red")
    plt.scatter(negative[:, 0], negative[:, 1], label="Negative", color="blue")
    plt.legend()
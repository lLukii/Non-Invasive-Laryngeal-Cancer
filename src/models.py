import torch.nn as nn
import torch.nn.functional as F
from cfg import *
from transformers import Wav2Vec2Model

class ContrastiveLoss(nn.Module):
  def __init__(self, reduction="mean", margin=0):
    super().__init__()
    self.reduction = reduction
    self.margin = margin
    self.loss = nn.CosineEmbeddingLoss(margin=self.margin, reduction=self.reduction)

  def forward(self, emb, y):
    positive = emb[y == 1]
    negative = emb[y == 0]
    return self.loss(positive, negative, -torch.ones(positive.shape[0], device=device))


class Wav2VecClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.wav2vec = Wav2Vec2Model.from_pretrained(model_name)
    self.embedding_space = nn.Sequential(
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(256, 128)
    )
    self.logit_space = nn.Linear(1024, 2)
    self.dropout = nn.Dropout(p=0.1)

  def forward(self, x, attn, return_features=False):
    x = self.wav2vec(input_values=x, attention_mask=attn).last_hidden_state
    x = self.dropout(x)
    x = torch.mean(x, dim=1)
    logits = self.logit_space(x)
    embeddings = F.normalize(self.embedding_space(x), dim=-1)
    return embeddings, logits


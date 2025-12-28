import torch.nn as nn
from torch.utils.data import Dataset, Sampler
import torchaudio.transforms as T
from transformers import Wav2Vec2FeatureExtractor
from cfg import *
import numpy as np

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

class AudioData(Dataset):
  def __init__(self, x, y, augment=False):
    self.x = x
    self.y = y
    self.augment = augment
    self.augmentations = nn.Sequential(
        T.Vol(gain=np.random.uniform(0.8, 1.2)),
        T.PitchShift(sr, n_steps=np.random.randint(-3, 3)),
        T.SpeedPerturbation(sr, [0.9, 1.0, 1.0, 1.0, 1.1]),
    ).to(device)
  def __len__(self): return len(self.x)
  def __getitem__(self, idx):
    if self.augment:
      with torch.no_grad():
        x1, _ = self.augmentations(torch.tensor(self.x[idx], device=device))
      processed_x1 = self.extract_signal(x1)
      return processed_x1, self.y[idx]

    return self.extract_signal(self.x[idx]), self.y[idx]

  def extract_signal(self, x):
    processed = feature_extractor(x, sampling_rate=sr, return_tensors="pt", padding="max_length", max_length=48000,
                                  device=device, return_attention_mask=True)
    input_features = processed.input_values
    attention_mask = processed.attention_mask
    if input_features.shape[1] > 48000:
      input_features = input_features[:, :48000]
      attention_mask = attention_mask[:, :48000]

    return input_features.squeeze(), attention_mask.squeeze()

class StratifiedSampler(Sampler):
  """
  Stratified Sampler to ensure proper training of minority class without distorting the true class distribution.
  Weighted Random Sampler doesn't do this well, in that sense, so we'll use this instead.
  """
  def __init__(self, labels, batch_size, min_positives):
    self.labels = labels
    self.batch_size = batch_size
    self.min_positives = min_positives
    self.num_batches = (len(labels) + self.batch_size - 1) // self.batch_size

    self.pos_ind = torch.where(self.labels == 1)[0]
    self.neg_ind = torch.where(self.labels == 0)[0]

  def __iter__(self):
    """
    Greedily assign each batch self.min_positive positive samples, and fill the remaining with negative ones.
    """
    shuffled_pos = torch.randperm(len(self.pos_ind))
    shuffled_neg = torch.randperm(len(self.neg_ind))
    pos_idx, neg_idx = 0, 0
    batches = []
    for _ in range(self.num_batches):
      for _ in range(self.min_positives):
        if pos_idx >= len(shuffled_pos):
          pos_idx = 0
          shuffled_pos = torch.randperm(len(self.pos_ind))
        batches.append(self.pos_ind[shuffled_pos[pos_idx]])
        pos_idx += 1

      for _ in range(self.batch_size - self.min_positives):
        if neg_idx >= len(shuffled_neg):
          neg_idx = 0
          shuffled_neg = torch.randperm(len(self.neg_ind))
        batches.append(self.neg_ind[shuffled_neg[neg_idx]])
        neg_idx += 1

    yield from batches

  def __len__(self):
    return len(self.labels)

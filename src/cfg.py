import torch

lr = 1e-4
batch_size = 32
sr = 16000
padding = 48000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
femh_path = "/content/drive/MyDrive/ISEF/Datasets/FEMH"
svd_path = "/content/drive/MyDrive/ISEF/Datasets/SVD"
model_name = "facebook/wav2vec2-large-xlsr-53"
num_epochs = 10
c_weight = 0.6
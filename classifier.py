from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler, random_split
import torch.nn as nn

def load_model():
    model = models.inception_v3(pretrained=False)
    n_classes = 120
    n_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(n_inputs, 1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, n_classes),
        nn.LogSoftmax(dim=1))
    weights = torch.load('weights/dog_inception.pt', map_location="cpu")
    model.load_state_dict(weights['state_dict'])
    mode.idx_to_class = weights['idx_to_class']
    
    return model
    
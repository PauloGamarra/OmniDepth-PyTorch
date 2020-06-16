from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import models

class DenseNetUprightAdjustment(nn.Module):
    def __init__(self):
        super(DenseNetUprightAdjustment, self).__init__()
        densenet = models.densenet121(pretrained=True)
        self.features = densenet.features
        self.n_features = densenet.classifier.in_features
        self.regressor = nn.Linear(93184, 2)
        self.parameters_list = list(self.parameters())

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.features(x)
        x = x.view(batch_size, -1)
        x = self.regressor(x)
        return x

import torch
import torch.nn as nn
import networkx as nx
from sklearn.datasets import fetch_openml
import scipy
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import TensorDataset, DataLoader




def cross_entropy_loss_with_l2(model, data, targets, l2_strength):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(model(data), targets)
    l2_reg = sum(param.pow(2).sum() for param in model.parameters())
    return loss + l2_strength * l2_reg
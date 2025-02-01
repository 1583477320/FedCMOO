# Adapted from LibMTL library

from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import numpy as np
import torch
import codecs
import copy
import torchvision
from torchvision import datasets, transforms
import sys
# Add the parent directory to the system path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from exp_utils import *
import logging

from torch.utils.data.dataset import Dataset
import torch.nn as nn
from torch.nn import GRU, Linear, ReLU, Sequential
import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv
from torch_geometric.nn.aggr import Set2Set
from torch_geometric.utils import remove_self_loops


def trainLoopPreprocess(batch):
    return batch

class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data

# class QM9Dataset(Dataset):
#     def __init__(self, dataset, target: list):
#         self.dataset = dataset
#         self.target = target

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         data = self.dataset.__getitem__(idx)
#         label = {}
#         for tn in self.target:
#             label[str(tn)] = data.y[:, tn]
#         return [data] + [v for v in label.values()] + [0] # 0 unique label not used, for consistency only

class QM9Dataset(Dataset):
    def __init__(self, dataset, target: list, device='cuda'):
        self.dataset = dataset
        self.target = target
        self.device = device

        # Move dataset to GPU during initialization
        self.data_list = []
        for data in self.dataset:
            # Move each graph to the desired device
            data.edge_index = data.edge_index.to(self.device)
            data.x = data.x.to(self.device)
            data.y = data.y.to(self.device)
            if data.edge_attr is not None:
                data.edge_attr = data.edge_attr.to(self.device)
            self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        label = {}
        for tn in self.target:
            label[str(tn)] = data.y[:, tn]
        return [data] + [v for v in label.values()] + [0]  # 0 unique label not used, for consistency

class Dataset(data.Dataset):
    def __init__(self, root=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'QM9'),
    # def __init__(self, root=os.path.join(os.getcwd(), '..', '..', 'data', 'QM9'),
                 split='train', pre_transform=True, target_transform=None, download=True, datasetdevice='cpu'):
        self.root = os.path.expanduser(root)
        self.train = split == 'train'
        self.split = split
        self.datasetdevice = datasetdevice
        self.pre_transform = pre_transform
        self.tasks = get_tasks()
        self.multi = True

        self.target = [int(i[1:]) for i in get_tasks()]

        if not os.path.exists(self.root):
            os.makedirs(self.root)


        transform = T.Compose([Complete(), T.Distance(norm=False)])
        dataset = QM9(self.root, transform=transform)
        
        # Normalize targets to mean = 0 and std = 1.
        mean = dataset.data.y.mean(dim=0, keepdim=True)
        std = dataset.data.y.std(dim=0, keepdim=True)
        dataset.data.y = (dataset.data.y - mean) / std
        split = torch.load(self.root+'/random_split.t')
        if self.split == 'train':
            self.qm9_dataset = QM9Dataset(dataset[split][20000:], self.target, device = datasetdevice)
            self.qm9_dataset.rawAll = dataset[split][20000:]
        elif self.split == 'val':
            self.qm9_dataset = QM9Dataset(dataset[split][10000:20000], self.target, device = datasetdevice)
        elif self.split == 'test':
            self.qm9_dataset = QM9Dataset(dataset[split][:10000], self.target, device = datasetdevice)
            
        
        # test_loader = DataLoader(test_dataset, batch_size=params.bs, shuffle=False, num_workers=2, pin_memory=True)
        # val_loader = DataLoader(val_dataset, batch_size=params.bs, shuffle=False, num_workers=2, pin_memory=True)
        # train_loader = DataLoader(train_dataset, batch_size=params.bs, shuffle=True, num_workers=2, pin_memory=True)


### Model ###

# define encoder and decoders
# class Rep(torch.nn.Module):
#     def __init__(self, dim=64):
#         super().__init__()
#         self.lin0 = torch.nn.Linear(11, dim)

#         nn = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
#         self.conv = NNConv(dim, dim, nn, aggr='mean')
#         self.gru = GRU(dim, dim)

#         self.set2set = Set2Set(dim, processing_steps=3)
#         self.lin1 = torch.nn.Linear(2 * dim, dim)
#         # self.lin2 = torch.nn.Linear(dim, 1)

#     def forward(self, data, mask = None):
#         out = F.relu(self.lin0(data.x))
#         h = out.unsqueeze(0)

#         for i in range(3):
#             m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
#             out, h = self.gru(m.unsqueeze(0), h)
#             out = out.squeeze(0)

#         out = self.set2set(out, data.batch)
#         out = F.relu(self.lin1(out))
#         # out = self.lin2(out)
#         return out , mask#.view(-1) 

class Rep(torch.nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.lin0 = torch.nn.Linear(11, dim)

        nn = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)

    def forward(self, data, mask=None):
        # Compact GRU weights to ensure contiguity
        self.gru.flatten_parameters()

        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        return out, mask

class Decoder(torch.nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.lin = torch.nn.Linear(64, 1)
    def forward(self, x, mask = None):
        return self.lin(x), mask

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_model(config) -> object:
    device = config['model_device']
    model = {}
    model['rep'] = Rep()
    model['rep'].to(device)
    for t in get_tasks():
        model[t] = Decoder()
        model[t].to(device)
    return model

def get_loss():
    loss_fn = {}
    for t in get_tasks():
        loss_fn[t] = torch.nn.MSELoss()
    return loss_fn

def get_tasks():
    return ['T'+str(i) for i in [0, 1, 2, 3, 5, 6, 12, 13, 14, 15, 11]]



if __name__ == '__main__':
    import torch
    import torchvision
    import matplotlib.pyplot as plt
    from torchvision import transforms
    dst = Dataset(train=True, download=True, transform=global_transformer(), multi=True)
    loader = torch.utils.data.DataLoader(dst, batch_size=10, shuffle=True, num_workers=10)
    for dat in loader:
        break


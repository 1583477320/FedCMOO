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
from exp_utils import *
import logging


### Data ###

def global_transformer():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])
def trainLoopPreprocess(batch):
    return batch

def trainLoopPreprocess(batch):
    N, C, H, W = batch.size()
    angles = torch.rand(N, device=batch.device) * 50 - 25  # Random angles between -30 and 30 degrees
    
    # Compute the rotation matrices for each angle
    theta = torch.zeros(N, 2, 3, device=batch.device)
    cos_vals = torch.cos(angles * torch.pi / 180.0)
    sin_vals = torch.sin(angles * torch.pi / 180.0)
    
    theta[:, 0, 0] = cos_vals
    theta[:, 0, 1] = -sin_vals
    theta[:, 1, 0] = sin_vals
    theta[:, 1, 1] = cos_vals
    grid = F.affine_grid(theta, batch.size(), align_corners=False)
    return F.grid_sample(batch, grid, align_corners=False)
    
class Dataset(data.Dataset):
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    multi_training_file = 'multi_training.pt'
    multi_test_file = 'multi_test.pt'

    def __init__(self, root=os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'MNIST_FMNIST'),
                 split='train', transform=global_transformer(), pre_transform=True, target_transform=None, download=True, multi=True, datasetdevice = 'cpu'):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        if split == 'train':
            self.train = True   # training set or test set
        elif split == 'test':
            self.train = False   # training set or test set
        self.split = split
        self.multi = multi
        self.datasetdevice = datasetdevice
        self.pre_transform = pre_transform
        self.tasks = get_tasks()

        # Ensure the MultiMNIST directory exists
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if not self._check_multi_exists():
            raise RuntimeError('Multi Task extension not found.' +
                               ' You can use download=True to download it')

        if multi:
            if self.train:
                self.data, self.labels_l, self.labels_r, self.labels_unique = torch.load(
                    os.path.join(self.root, 'MNIST_FMNIST', self.processed_folder, self.multi_training_file))
            else:
                self.data, self.labels_l, self.labels_r, self.labels_unique = torch.load(
                    os.path.join(self.root, 'MNIST_FMNIST', self.processed_folder, self.multi_test_file))
        else:
            if self.train:
                self.data, self.labels = torch.load(
                    os.path.join(self.root, 'MNIST', self.processed_folder, self.training_file))
            else:
                self.data, self.labels = torch.load(
                    os.path.join(self.root, 'MNIST', self.processed_folder, self.test_file))

        if self.pre_transform:            
            logging.info('Pre-transforming the dataset...')
            self.data = self.pretransform_data(self.data).to(self.datasetdevice)
            if self.multi:
                self.labels_l = self.labels_l.to(self.datasetdevice)
                self.labels_r = self.labels_r.to(self.datasetdevice)
                self.labels_unique = self.labels_unique.to(self.datasetdevice)
            else:
                self.labels = self.labels.to(self.datasetdevice)

    def __getitem__(self, index):
        if self.multi:
            img, target_l, target_r, target_unique = self.data[index], self.labels_l[index], self.labels_r[index], self.labels_unique[index]
        else:
            img, target = self.data[index], self.labels[index]


        if not self.pre_transform and self.transform is not None:
            img = Image.fromarray(img.numpy().astype(np.uint8), mode='L')
            img = self.transform(img)

        if not self.pre_transform and self.target_transform is not None:
            if self.multi:
                target_l, target_r = self.target_transform(target_l), self.target_transform(target_r)
            else:
                target = self.target_transform(target)

        if self.multi:
            return img, target_l, target_r, target_unique # first: input, ...targets as many as #objectives, unique targets for distribution purpose
        else:
            return img, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, 'MNIST', self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, 'MNIST', self.processed_folder, self.test_file)) and \
            os.path.exists(os.path.join(self.root, 'FashionMNIST', self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, 'FashionMNIST', self.processed_folder, self.test_file))
    
    def _check_multi_exists(self):
        return os.path.exists(os.path.join(self.root, 'MNIST_FMNIST', self.processed_folder, self.multi_training_file)) and \
            os.path.exists(os.path.join(self.root, 'MNIST_FMNIST', self.processed_folder, self.multi_test_file))

    def download(self):
        """Download the MNIST data using torchvision if it doesn't exist in processed_folder already."""
        if self._check_exists() and self._check_multi_exists():
            return

        # Download the MNIST dataset using torchvision
        transform = transforms.Compose([transforms.ToTensor()])
        train_datasets = [datasets.MNIST(self.root, train=True, download=True, transform=transform), 
                          datasets.FashionMNIST(self.root, train=True, download=True, transform=transform)]
        test_datasets = [datasets.MNIST(self.root, train=False, download=True, transform=transform), 
                          datasets.FashionMNIST(self.root, train=False, download=True, transform=transform)]

        train_data_list = [train_dataset.data for train_dataset in train_datasets]
        train_labels_list = [train_dataset.targets for train_dataset in train_datasets]
        test_data_list = [test_dataset.data for test_dataset in test_datasets]
        test_labels_list = [test_dataset.targets for test_dataset in test_datasets]

        # Process and save as torch files
        print('Processing...')
        mnist_training_set = (train_data_list[0], train_labels_list[0])
        mnist_test_set = (test_data_list[0], test_labels_list[0])
        if not os.path.exists(os.path.join(self.root, 'MNIST', self.processed_folder)):
            os.makedirs(os.path.join(self.root, 'MNIST', self.processed_folder))
        with open(os.path.join(self.root, 'MNIST', self.processed_folder, self.training_file), 'wb') as f:
            torch.save(mnist_training_set, f)
        with open(os.path.join(self.root, 'MNIST', self.processed_folder, self.test_file), 'wb') as f:
            torch.save(mnist_test_set, f)

        fmnist_training_set = (train_data_list[1], train_labels_list[1])
        fmnist_test_set = (test_data_list[1], test_labels_list[1])
        if not os.path.exists(os.path.join(self.root, 'FashionMNIST', self.processed_folder)):
            os.makedirs(os.path.join(self.root, 'FashionMNIST', self.processed_folder))
        with open(os.path.join(self.root, 'FashionMNIST', self.processed_folder, self.training_file), 'wb') as f:
            torch.save(fmnist_training_set, f)
        with open(os.path.join(self.root, 'FashionMNIST', self.processed_folder, self.test_file), 'wb') as f:
            torch.save(fmnist_test_set, f)
        
        self.create_multi_task_data(mnist_train_data = train_data_list[0], 
                                    mnist_train_labels = train_labels_list[0], 
                                    fashion_train_data = train_data_list[1],
                                    fashion_train_labels = train_labels_list[1],
                                    mnist_test_data = test_data_list[0], 
                                    mnist_test_labels = test_labels_list[0], 
                                    fashion_test_data = test_data_list[1],
                                    fashion_test_labels = test_labels_list[1])

        print('Done!')

    def create_multi_task_data(self, mnist_train_data, mnist_train_labels, fashion_train_data, fashion_train_labels, mnist_test_data, mnist_test_labels, fashion_test_data, fashion_test_labels):
        # Train dataset
        combined_train_data, combined_train_labels_l, combined_train_labels_r, combined_train_labels_unique = self.generate_multi_task_data(
            mnist_train_data, mnist_train_labels, fashion_train_data, fashion_train_labels)

        # Test dataset
        combined_test_data, combined_test_labels_l, combined_test_labels_r, combined_test_labels_unique = self.generate_multi_task_data(
            mnist_test_data, mnist_test_labels, fashion_test_data, fashion_test_labels)

        # Save train and test datasets
        combined_training_set = (combined_train_data, combined_train_labels_l, combined_train_labels_r, combined_train_labels_unique)
        combined_test_set = (combined_test_data, combined_test_labels_l, combined_test_labels_r, combined_test_labels_unique)

        if not os.path.exists(os.path.join(self.root, 'MNIST_FMNIST', self.processed_folder)):
            os.makedirs(os.path.join(self.root, 'MNIST_FMNIST', self.processed_folder))
        with open(os.path.join(self.root, 'MNIST_FMNIST', self.processed_folder, self.multi_training_file), 'wb') as f:
            torch.save(combined_training_set, f)
        with open(os.path.join(self.root, 'MNIST_FMNIST', self.processed_folder, self.multi_test_file), 'wb') as f:
            torch.save(combined_test_set, f)

    def generate_multi_task_data(self, mnist_data, mnist_labels, fashion_data, fashion_labels):
        length = len(mnist_labels)
        num_rows, num_cols = mnist_data.size(1), mnist_data.size(2)
        combined_data = torch.zeros((length, num_rows, num_cols), dtype=torch.uint8)
        combined_labels_l = torch.zeros((length), dtype=torch.long)
        combined_labels_r = torch.zeros((length), dtype=torch.long)
        unique_labels = torch.zeros((length), dtype=torch.long)
        
        # Create a permuted list of indices for FashionMNIST
        fashion_indices = torch.randperm(len(fashion_data))
        fashion_index = 0
        
        for left in range(length):
            # Left image: MNIST
            mnist_img = mnist_data[left]
            
            # Right image: FashionMNIST
            if fashion_index >= len(fashion_data):  # If we run out of indices, reshuffle
                fashion_indices = torch.randperm(len(fashion_data))
                fashion_index = 0
            
            fashion_img = fashion_data[fashion_indices[fashion_index]]
            fashion_index += 1
    
            # Create combined image: MNIST on left, FashionMNIST on right
            lsize = 42
            new_im = torch.zeros((lsize, lsize), dtype=torch.uint8)
            new_im[0:28, 0:28] = mnist_img
            new_im[lsize-28:lsize, lsize-28:lsize] = fashion_img
            new_im[lsize-28:28, lsize-28:28] = torch.max(mnist_img[lsize-28:28, lsize-28:28], fashion_img[0:int(2*28-lsize), 0:int(2*28-lsize)])
            
            # Resize back to 28x28 if needed
            combined_data_im = transforms.functional.resize(new_im.unsqueeze(0), (28, 28), interpolation=transforms.InterpolationMode.NEAREST)
            combined_data[left] = combined_data_im.squeeze(0)  # Remove channel dimension after resize
            
            # Labels
            combined_labels_l[left] = mnist_labels[left]
            combined_labels_r[left] = fashion_labels[fashion_indices[fashion_index-1]]
            unique_labels[left] = generate_unique_label(mnist_labels[left].item(), fashion_labels[fashion_indices[fashion_index-1]].item())
        
        return combined_data, combined_labels_l, combined_labels_r, unique_labels

    def pretransform_data(self, data):
        transformed_data = []
        for img in data:
            img = Image.fromarray(img.numpy().astype(np.uint8), mode='L')
            img = self.transform(img)
            transformed_data.append(img)
        return torch.stack(transformed_data)


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

def generate_unique_label(left_label, right_label):
    return left_label * 10 + right_label

def split_val_dataset(dataset, val_ratio, val_seed = None):
    """
    Split the dataset into training and validation sets while preserving attributes.
    Args:
        dataset (Dataset): The original dataset to split.
        val_ratio (float): The ratio of the validation set size to the total dataset size.
    Returns:
        train_dataset (Dataset): The training subset.
        val_dataset (Dataset): The validation subset.
    """
    # Determine the sizes of the training and validation sets
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    
    # Generate indices for the split
    indices = list(range(len(dataset)))
    if val_seed:
        np.random.seed(val_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:train_size], indices[train_size:]
    
    # Create copies of the dataset for training and validation sets
    train_dataset = copy.deepcopy(dataset)
    val_dataset = copy.deepcopy(dataset)
    
    # Assign the subsets of data and labels to the new datasets
    if dataset.multi:
        train_dataset.data = dataset.data[train_indices]
        train_dataset.labels_l = dataset.labels_l[train_indices]
        train_dataset.labels_r = dataset.labels_r[train_indices]
        train_dataset.labels_unique = dataset.labels_unique[train_indices]

        val_dataset.data = dataset.data[val_indices]
        val_dataset.labels_l = dataset.labels_l[val_indices]
        val_dataset.labels_r = dataset.labels_r[val_indices]
        val_dataset.labels_unique = dataset.labels_unique[val_indices]
    else:
        train_dataset.data = dataset.data[train_indices]
        train_dataset.labels = dataset.labels[train_indices]

        val_dataset.data = dataset.data[val_indices]
        val_dataset.labels = dataset.labels[val_indices]

    # Preserve other attributes
    for attr in ['root', 'transform', 'target_transform', 'train', 'split', 'multi', 'datasetdevice', 'pre_transform']:
        setattr(train_dataset, attr, getattr(dataset, attr))
        setattr(val_dataset, attr, getattr(dataset, attr))

    # Update the 'train' attribute
    train_dataset.train = True
    val_dataset.train = False
    
    if val_seed:
        np.random.seed()
    return train_dataset, val_dataset


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

def generate_unique_label(left_label, right_label):
    return left_label * 10 + right_label

def split_val_dataset(dataset, val_ratio, val_seed = None):
    """
    Split the dataset into training and validation sets while preserving attributes.
    Args:
        dataset (Dataset): The original dataset to split.
        val_ratio (float): The ratio of the validation set size to the total dataset size.
    Returns:
        train_dataset (Dataset): The training subset.
        val_dataset (Dataset): The validation subset.
    """
    # Determine the sizes of the training and validation sets
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    
    # Generate indices for the split
    indices = list(range(len(dataset)))
    if val_seed:
        np.random.seed(val_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:train_size], indices[train_size:]
    
    # Create copies of the dataset for training and validation sets
    train_dataset = copy.deepcopy(dataset)
    val_dataset = copy.deepcopy(dataset)
    
    # Assign the subsets of data and labels to the new datasets
    if dataset.multi:
        train_dataset.data = dataset.data[train_indices]
        train_dataset.labels_l = dataset.labels_l[train_indices]
        train_dataset.labels_r = dataset.labels_r[train_indices]
        train_dataset.labels_unique = dataset.labels_unique[train_indices]

        val_dataset.data = dataset.data[val_indices]
        val_dataset.labels_l = dataset.labels_l[val_indices]
        val_dataset.labels_r = dataset.labels_r[val_indices]
        val_dataset.labels_unique = dataset.labels_unique[val_indices]
    else:
        train_dataset.data = dataset.data[train_indices]
        train_dataset.labels = dataset.labels[train_indices]

        val_dataset.data = dataset.data[val_indices]
        val_dataset.labels = dataset.labels[val_indices]

    # Preserve other attributes
    for attr in ['root', 'transform', 'target_transform', 'train', 'split', 'multi', 'datasetdevice', 'pre_transform']:
        setattr(train_dataset, attr, getattr(dataset, attr))
        setattr(val_dataset, attr, getattr(dataset, attr))

    # Update the 'train' attribute
    train_dataset.train = True
    val_dataset.train = False
    
    if val_seed:
        np.random.seed()
    return train_dataset, val_dataset


### Model ###
# From https://github.com/isl-org/MultiObjectiveOptimization

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MultiLeNetR(nn.Module):
    def __init__(self):
        super(MultiLeNetR, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc = nn.Linear(320, 50)

    def dropout2dwithmask(self, x, mask):
        channel_size = x.shape[1]
        if mask is None:
            mask = Variable(torch.bernoulli(torch.ones(1, channel_size, 1, 1) * 0.5).to(x.device))
        # mask = mask.expand_as(x)
        mask = mask.expand(x.shape)
        return mask
    def dropout2dwithmask(self, x, mask):
        channel_size = x.shape[1]
        if mask is None:
            mask = Variable(torch.bernoulli(torch.ones(1, channel_size, 1, 1) * 0.5).to(x.device))
        mask = mask.expand(x.shape)
        return mask
        
    def forward(self, x, mask):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2(x)
        mask = self.dropout2dwithmask(x, mask)
        if self.training:
            x = x*mask
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc(x))
        return x, mask

class MultiLeNetR(nn.Module):
    def __init__(self):
        super(MultiLeNetR, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding = 1)
        self.conv2 = nn.Conv2d(10, 15, kernel_size=3, padding =0)
        self.conv2_drop = nn.Dropout2d()
        self.fc = nn.Linear(36*15, 50)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x, mask):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 36*15)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        return x, mask

class MultiLeNetO(nn.Module):
    def __init__(self):
        super(MultiLeNetO, self).__init__()
        self.fc1 = nn.Linear(50, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x, mask):
        x = F.relu(self.fc1(x))
        if mask is None:
            mask = Variable(torch.bernoulli(x.data.new(x.data.size()).fill_(0.5)))        
        if self.training:
            x = x*mask
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), mask
class MultiLeNetO(nn.Module):
    def __init__(self):
        super(MultiLeNetO, self).__init__()
        self.fc1 = nn.Linear(50, 50)
        self.fc2 = nn.Linear(50, 10)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x, mask):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), mask

def get_model(config) -> object:
    device = config['model_device']
    model = {}
    model['rep'] = MultiLeNetR()
    model['rep'].to(device)
    model['L'] = MultiLeNetO()
    model['L'].to(device)
    model['R'] = MultiLeNetO()
    model['R'].to(device)
    return model

def get_loss():
    loss_fn = {}
    for t in ['L', 'R']:
        loss_fn[t] = nll
    return loss_fn

def get_tasks():
    return ['L', 'R']


if __name__ == '__main__':
    import torch
    import torchvision
    import matplotlib.pyplot as plt
    from torchvision import transforms
    dst = Dataset(split = 'train', download=True, transform=global_transformer(), multi=True)
    loader = torch.utils.data.DataLoader(dst, batch_size=10, shuffle=True, num_workers=10)
    for dat in loader:
        ims = dat[0].view(10, 28, 28).numpy()
        labs_l = dat[1]
        labs_r = dat[2]
        labs_unique = dat[3]
        f, axarr = plt.subplots(2, 5)
        for j in range(5):
            for i in range(2):
                axarr[i][j].imshow(ims[j * 2 + i, :, :], cmap='gray')
                axarr[i][j].set_title('L:{} R:{} U:{}'.format(labs_l[j * 2 + i], labs_r[j * 2 + i], labs_unique[j * 2 + i]))
        plt.show()
        a = input()
        if a == 'ex':
            break
        else:
            plt.close()


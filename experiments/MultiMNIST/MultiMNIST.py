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

    def __init__(self, root=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'MultiMNIST'),
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
                    os.path.join(self.root, 'MNIST', self.processed_folder, self.multi_training_file))
            else:
                self.data, self.labels_l, self.labels_r, self.labels_unique = torch.load(
                    os.path.join(self.root, 'MNIST', self.processed_folder, self.multi_test_file))
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
            os.path.exists(os.path.join(self.root, 'MNIST', self.processed_folder, self.test_file))
    
    def _check_multi_exists(self):
        return os.path.exists(os.path.join(self.root, 'MNIST', self.processed_folder, self.multi_training_file)) and \
            os.path.exists(os.path.join(self.root, 'MNIST', self.processed_folder, self.multi_test_file))

    def download(self):
        """Download the MNIST data using torchvision if it doesn't exist in processed_folder already."""
        if self._check_exists() and self._check_multi_exists():
            return

        # Download the MNIST dataset using torchvision
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST(self.root, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(self.root, train=False, download=True, transform=transform)

        train_data = train_dataset.data
        train_labels = train_dataset.targets
        test_data = test_dataset.data
        test_labels = test_dataset.targets

        # Process and save as torch files
        print('Processing...')
        mnist_training_set = (train_data, train_labels)
        mnist_test_set = (test_data, test_labels)
        if not os.path.exists(os.path.join(self.root, 'MNIST', self.processed_folder)):
            os.makedirs(os.path.join(self.root, 'MNIST', self.processed_folder))
        with open(os.path.join(self.root, 'MNIST', self.processed_folder, self.training_file), 'wb') as f:
            torch.save(mnist_training_set, f)
        with open(os.path.join(self.root, 'MNIST', self.processed_folder, self.test_file), 'wb') as f:
            torch.save(mnist_test_set, f)
        
        self.create_multi_task_data(train_data, train_labels, test_data, test_labels)

        print('Done!')

    def create_multi_task_data(self, train_data, train_labels, test_data, test_labels):
        multi_train_data, multi_train_labels_l, multi_train_labels_r, multi_train_labels_unique = self.generate_multi_task_data(train_data, train_labels)
        multi_test_data, multi_test_labels_l, multi_test_labels_r, multi_test_labels_unique = self.generate_multi_task_data(test_data, test_labels)

        multi_mnist_training_set = (multi_train_data, multi_train_labels_l, multi_train_labels_r, multi_train_labels_unique)
        multi_mnist_test_set = (multi_test_data, multi_test_labels_l, multi_test_labels_r, multi_test_labels_unique)

        with open(os.path.join(self.root, 'MNIST', self.processed_folder, self.multi_training_file), 'wb') as f:
            torch.save(multi_mnist_training_set, f)
        with open(os.path.join(self.root, 'MNIST', self.processed_folder, self.multi_test_file), 'wb') as f:
            torch.save(multi_mnist_test_set, f)

    def generate_multi_task_data(self, data, labels):
        length = len(labels)
        num_rows, num_cols = data.size(1), data.size(2)
        multi_length = length * 1
        multi_data = torch.zeros((1 * length, num_rows, num_cols), dtype=torch.uint8)
        multi_labels_l = torch.zeros((1 * length), dtype=torch.long)
        multi_labels_r = torch.zeros((1 * length), dtype=torch.long)
        unique_labels = torch.zeros((1 * length), dtype=torch.long)
        extension = torch.zeros(1 * length, dtype=torch.int32)
        for left in range(length):
            chosen_ones = torch.randperm(length)[:1]
            extension[left * 1:(left + 1) * 1] = chosen_ones
            for j, right in enumerate(chosen_ones):
                lim = data[left]
                rim = data[right]
                new_im = torch.zeros((36, 36), dtype=torch.uint8)
                new_im[0:28, 0:28] = lim
                new_im[6:34, 6:34] = rim
                new_im[6:28, 6:28] = torch.max(lim[6:28, 6:28], rim[0:22, 0:22])
                new_im = new_im.unsqueeze(0)  # Add channel dimension
                multi_data_im = transforms.functional.resize(new_im, (28, 28), interpolation=transforms.InterpolationMode.NEAREST)
                multi_data[left * 1 + j] = multi_data_im.squeeze(0)  # Remove channel dimension after resize
                multi_labels_l[left * 1 + j] = labels[left]
                multi_labels_r[left * 1 + j] = labels[right]
                unique_labels[left * 1 + j] = generate_unique_label(labels[left].item(), labels[right].item())
        return multi_data, multi_labels_l, multi_labels_r, unique_labels

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

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)  # normal: mean=0, std=1
            elif isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight.data)  # normal: mean=0, std=1

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

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)  # normal: mean=0, std=1
            elif isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight.data)  # normal: mean=0, std=1

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


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

def global_transformer(split = 'train'):
    if split == 'train':
        return transforms.Compose([
            transforms.RandomRotation(30), # Rotation from https://github.com/salomonhotegni/MDMTN/blob/main/Train_and_Test.py
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    elif split == 'test':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


def trainLoopPreprocess(batch):
    N, C, H, W = batch.size()

    # Step 1: Random flip on x-axis with probability 0.3
    if torch.rand(1).item() < 0.4:
        batch = torch.flip(batch, dims=[3])  # Flip on x-axis (width axis)
    return batch


class Dataset(data.Dataset):
    def __init__(self, root=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'CIFAR10_MNIST'),
                 split='train', pre_transform=True, target_transform=None, download=True, datasetdevice='cpu'):
        self.root = os.path.expanduser(root)
        self.transform = global_transformer(split = split)
        self.target_transform = target_transform
        self.train = split == 'train'
        self.split = split
        self.datasetdevice = datasetdevice
        self.pre_transform = pre_transform
        self.tasks = get_tasks()
        self.multi = True

        # Ensure the CIFAR10_MNIST directory exists
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        if download:
            self.download()

        # Load combined dataset if it exists
        if self._check_combined_exists():
            self.data, self.labels_cifar, self.labels_mnist, self.labels_unique = torch.load(
                os.path.join(self.root, 'combined_{}.pt'.format('train' if self.train else 'test')))
        else:
            if self.train:
                self.cifar_data, self.cifar_labels = torch.load(os.path.join(self.root, 'cifar_training.pt'))
                self.mnist_data, self.mnist_labels = torch.load(os.path.join(self.root, 'mnist_training.pt'))
            else:
                self.cifar_data, self.cifar_labels = torch.load(os.path.join(self.root, 'cifar_test.pt'))
                self.mnist_data, self.mnist_labels = torch.load(os.path.join(self.root, 'mnist_test.pt'))

            self.data, self.labels_cifar, self.labels_mnist, self.labels_unique = self.create_combined_data()

            # Save the combined dataset for future use
            torch.save((self.data, self.labels_cifar, self.labels_mnist, self.labels_unique),
                       os.path.join(self.root, 'combined_{}.pt'.format('train' if self.train else 'test')))

        if self.pre_transform:
            logging.info('Pre-transforming the dataset...')
            self.data = self.pretransform_data(self.data).to(self.datasetdevice)
            self.labels_cifar = self.labels_cifar.to(self.datasetdevice)
            self.labels_mnist = self.labels_mnist.to(self.datasetdevice)
            self.labels_unique = self.labels_unique.to(self.datasetdevice)
            self.rr = transforms.RandomRotation(30)

    def __getitem__(self, index):
        img, target_cifar, target_mnist, target_unique = self.data[index], self.labels_cifar[index], self.labels_mnist[index], self.labels_unique[index]

        if not self.pre_transform:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target_cifar = self.target_transform(target_cifar)
                target_mnist = self.target_transform(target_mnist)
        else:
            # img = self.rr(img)
            img = torchvision.transforms.functional.rotate(img, angle=np.random.randint(-30,31)) if self.split == 'train' else img  # Apply RandomRotation even if pre_transform is True
        return img, target_cifar, target_mnist, target_unique

    def pretransform_data(self, data):
        transformed_data = []
        normalization_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        for img in data:
            img = transforms.ToPILImage()(img)
            img = normalization_transform(img)  # Apply normalization only, excluding RandomRotation
            transformed_data.append(img)
        return torch.stack(transformed_data)

    def __len__(self):
        return len(self.data)

    def _check_combined_exists(self):
        return os.path.exists(os.path.join(self.root, 'combined_{}.pt'.format('train' if self.train else 'test')))

    def download(self):
        """Download the CIFAR-10 and MNIST data using torchvision if it doesn't exist in processed_folder already."""
        if os.path.exists(os.path.join(self.root, 'cifar_training.pt')) and \
           os.path.exists(os.path.join(self.root, 'cifar_test.pt')) and \
           os.path.exists(os.path.join(self.root, 'mnist_training.pt')) and \
           os.path.exists(os.path.join(self.root, 'mnist_test.pt')):
            return

        # Download the CIFAR-10 dataset using torchvision
        cifar_transform = transforms.Compose([transforms.ToTensor()])
        cifar_train_dataset = datasets.CIFAR10(self.root, train=True, download=True, transform=cifar_transform)
        cifar_test_dataset = datasets.CIFAR10(self.root, train=False, download=True, transform=cifar_transform)

        cifar_train_data = torch.stack([cifar_train_dataset[i][0] for i in range(len(cifar_train_dataset))])
        cifar_train_labels = torch.tensor(cifar_train_dataset.targets)
        cifar_test_data = torch.stack([cifar_test_dataset[i][0] for i in range(len(cifar_test_dataset))])
        cifar_test_labels = torch.tensor(cifar_test_dataset.targets)

        # Download the MNIST dataset using torchvision
        mnist_transform = transforms.Compose([transforms.ToTensor()])
        mnist_train_dataset = datasets.MNIST(self.root, train=True, download=True, transform=mnist_transform)
        mnist_test_dataset = datasets.MNIST(self.root, train=False, download=True, transform=mnist_transform)

        mnist_train_data = mnist_train_dataset.data
        mnist_train_labels = mnist_train_dataset.targets
        mnist_test_data = mnist_test_dataset.data
        mnist_test_labels = mnist_test_dataset.targets

        # Save datasets
        torch.save((cifar_train_data, cifar_train_labels), os.path.join(self.root, 'cifar_training.pt'))
        torch.save((cifar_test_data, cifar_test_labels), os.path.join(self.root, 'cifar_test.pt'))
        torch.save((mnist_train_data, mnist_train_labels), os.path.join(self.root, 'mnist_training.pt'))
        torch.save((mnist_test_data, mnist_test_labels), os.path.join(self.root, 'mnist_test.pt'))

    def create_combined_data(self):
        from PIL import ImageEnhance
        combined_data = []
        combined_labels_cifar = []
        combined_labels_mnist = []
        combined_labels_unique = []
    
        for i in range(len(self.cifar_data)):
            cifar_img = self.cifar_data[i]
            mnist_img = self.mnist_data[i % len(self.mnist_data)]  # cycle through MNIST data
    
            # Convert the CIFAR image to a PIL image
            pil_image = transforms.ToPILImage()(cifar_img)
            
            # Convert the MNIST image to a PIL image (no resizing here)
            mnist_image = transforms.ToPILImage()(mnist_img.unsqueeze(0))
            
            # Create a new RGB image with CIFAR image
            combined_image = pil_image.copy()
            
            # Center position calculation for MNIST image on CIFAR image
            mnist_size = mnist_image.size[0]
            cifar_size = pil_image.size[0]
            left = (cifar_size - mnist_size) // 2
            top = (cifar_size - mnist_size) // 2
            
            # Create an alpha channel mask with the white pixels having a slightly transparent alpha value
            alpha_mask = mnist_image.convert("L").point(lambda x: 156 if x > 0 else 0)
            
            # Paste the MNIST image onto the new image, centered
            combined_image.paste(mnist_image, (left, top), mask=alpha_mask)
            
            # Convert the combined image back to a tensor
            combined_tensor = transforms.ToTensor()(combined_image)
            
            combined_data.append(combined_tensor)
            combined_labels_cifar.append(self.cifar_labels[i])
            combined_labels_mnist.append(self.mnist_labels[i % len(self.mnist_labels)])
            combined_labels_unique.append(self.cifar_labels[i] * 10 + self.mnist_labels[i % len(self.mnist_labels)])
    
        combined_data = torch.stack(combined_data)
        combined_labels_cifar = torch.tensor(combined_labels_cifar)
        combined_labels_mnist = torch.tensor(combined_labels_mnist)
        combined_labels_unique = torch.tensor(combined_labels_unique)
    
        return combined_data, combined_labels_cifar, combined_labels_mnist, combined_labels_unique


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
        train_dataset.labels_cifar = dataset.labels_cifar[train_indices]
        train_dataset.labels_mnist = dataset.labels_mnist[train_indices]
        train_dataset.labels_unique = dataset.labels_unique[train_indices]

        val_dataset.data = dataset.data[val_indices]
        val_dataset.labels_cifar = dataset.labels_cifar[val_indices]
        val_dataset.labels_mnist = dataset.labels_mnist[val_indices]
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

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#FedDyn CIFAR-10 model
# class Rep(nn.Module):
#     def __init__(self):
#         super(Rep,self).__init__()
#         self.n_cls = 10
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(64*5*5, 384)
#         self.fc2 = nn.Linear(384, 192)
#
#     def forward(self, x, mask):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 64*5*5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return x, mask
#
#
# class Rep(nn.Module):
#     def __init__(self):
#         super(Rep,self).__init__()
#         self.n_cls = 10
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(64*5*5, 384)
#         self.fc2 = nn.Linear(384, 192)
#
#     def forward(self, x, mask):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 64*5*5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return x, mask
#
# class Rep(nn.Module):
#     def __init__(self):
#         super(Rep,self).__init__()
#         self.n_cls = 10
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(64*5*5, 384)
#         self.fc2 = nn.Linear(384, 192)
#
#     def forward(self, x, mask):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 64*5*5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return x, mask
#
# class Rep(nn.Module):
#     def __init__(self):
#         super(Rep,self).__init__()
#         self.n_cls = 10
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=3, padding = 1)
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding = 1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(64*8*8, 384)
#         self.fc2 = nn.Linear(384, 192)
#
#     def forward(self, x, mask):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 64*8*8)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return x, mask
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

class Rep(nn.Module):
    def __init__(self):
        super(Rep, self).__init__()
        self.n_cls = 10
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 384) 
        self.fc2 = nn.Linear(384, 192)
        
        # Dropout layer with a probability of 0.5 (can be adjusted)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, mask):
        # Convolutional layers with pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the tensor
        x = x.view(-1, 64 * 8 * 8)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after first fully connected layer
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # Apply dropout after second fully connected layer

        return x, mask

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data)  # normal: mean=0, std=1


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(192, 100)
        self.out = nn.Linear(100, 10)
        # self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x, mask):
        x = self.out(self.linear(x))
        # x = self.dropout(x)
        out = F.log_softmax(x, dim=1)
        return out, mask
    
    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data)  # normal: mean=0, std=1


def get_model(config) -> object:
    device = config['model_device']
    model = {}
    model['rep'] = Rep()
    model['rep'].to(device)
    model['C'] = Decoder()
    model['C'].to(device)
    model['M'] = Decoder()
    model['M'].to(device)
    return model

def get_loss():
    loss_fn = {}
    for t in ['C', 'M']:
        loss_fn[t] = nll
    return loss_fn

def get_tasks():
    return ['C', 'M']


if __name__ == '__main__':
    import torch
    import torchvision
    import matplotlib.pyplot as plt
    from torchvision import transforms
    dst = Dataset(train=True, download=True, transform=global_transformer(), multi=True)
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


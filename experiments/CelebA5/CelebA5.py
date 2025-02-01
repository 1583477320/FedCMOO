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
import re
import glob
import zipfile
import random



### Data ###

# Define the mean and image size as constants
device = 'cuda'
MEAN = np.array([73.15835921, 82.90891754, 72.39239876])
IMG_SIZE = (32, 32)
DATA_ZIP_FILE = 'celeba.zip'
TASK_GROUP_INDEX_SEED = 64

def global_transformer():
    """Global transformer function
    Mean subtraction, remap to [0,1], channel order transpose to make Torch happy
    """
    def transform_img(img):
        img = img.resize(IMG_SIZE, Image.BILINEAR)
        img = np.array(img)[:, :, ::-1]  # Convert PIL image to numpy array and reverse channels
        img = img.astype(np.float64)
        img -= MEAN  # Subtract mean
        img = img.astype(float) / 255.0  # Normalize to [0, 1]
        img = img[..., [2,1,0]]
        img = img.transpose(2, 0, 1)  # NHWC -> NCHW
        img = torch.from_numpy(img).float()  # Convert to torch tensor
        return img

    return transforms.Lambda(transform_img)
def trainLoopPreprocess(batch):
    return batch
    
class Dataset(data.Dataset):
    def __init__(self, root=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'CelebA'),
                 split='train', transform=global_transformer(), pre_transform=True, target_transform=None, download=True, multi=True, datasetdevice='cpu', faster_load=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # training set, val set, or test set
        self.multi = multi
        self.datasetdevice = datasetdevice
        self.pre_transform = pre_transform
        self.tasks = get_tasks()
        self.faster_load = faster_load

        # Ensure the CelebA directory exists
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        if download:
            self.download()

        if not self._check_exists() and not self._pretransformed_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        if self._check_exists():
            self._load_data()

        if self.pre_transform:
            self._pre_transform_data()

        self.original_labels = self.labels.clone()

        self.split_labels()        

    def _load_data(self):
        self.label_file = os.path.join(self.root, "list_attr_celeba.csv")
        label_map = {}
        with open(self.label_file, 'r') as l_file:
            labels = l_file.read().split('\n')[1:-1]
        for label_line in labels:
            f_name = label_line.split(',')[0].strip()
            # Ensure the filename has the correct extension
            if not f_name.endswith('.jpg'):
                f_name += '.jpg'
            label_txt = list(map(lambda x: int(x), re.sub('-1', '0', label_line).split(',')[1:]))
            label_map[f_name] = label_txt
    
        self.all_files = glob.glob(os.path.join(self.root, 'img_align_celeba/img_align_celeba/*.jpg'))
        if len(self.all_files) == 0:
            raise RuntimeError('No images found in img_align_celeba folder.')
    
        with open(os.path.join(self.root, 'list_eval_partition.csv'), 'r') as f:
            self.fl = fl = f.read().split('\n')[1:]  # Skip the header row
            fl.pop()  # Remove the last empty element if exists
            if 'train' in self.split:
                selected_files = list(filter(lambda x: x.split(',')[1].strip() == '0', fl))
            elif 'val' in self.split:
                selected_files = list(filter(lambda x: x.split(',')[1].strip() == '1', fl))
            elif 'test' in self.split:
                selected_files = list(filter(lambda x: x.split(',')[1].strip() == '2', fl))
            selected_file_names = list(map(lambda x: x.split(',')[0].strip(), selected_files))
    
        base_path = '/'.join(self.all_files[0].split('/')[:-1])
        self.files = list(map(lambda x: os.path.join(base_path, x), set(map(lambda x: x.split('/')[-1], self.all_files)).intersection(set(selected_file_names))))
        self.labels = torch.tensor(list(map(lambda x: label_map[x], set(map(lambda x: x.split('/')[-1], self.all_files)).intersection(set(selected_file_names)))))
        self.labels_unique = self.generate_unique_labels(self.labels)
    
        if len(self.files) < 2:
            raise Exception("No files for split=[%s] found in %s" % (self.split, self.root))
    
        print("Found %d %s images" % (len(self.files), self.split))

    def _pre_transform_data(self):
        if self.faster_load:
            pretransformed_data_file = os.path.join(self.root, f'pretransformed_{self.split}.pt')
            pretransformed_labels_file = os.path.join(self.root, f'pretransformed_labels_{self.split}.pt')
            if os.path.exists(pretransformed_data_file) and os.path.exists(pretransformed_labels_file):
                self.data = torch.load(pretransformed_data_file)
                self.labels = torch.load(pretransformed_labels_file)
                if not self._check_exists():
                    self.labels_unique = self.generate_unique_labels(self.labels)
            else:
                logging.info('Pre-transforming the dataset...')
                self.data = self.pretransform_data(self.files)
                self.labels = self.labels
                torch.save(self.data, pretransformed_data_file)
                torch.save(self.labels, pretransformed_labels_file)
        else:
            logging.info('Pre-transforming the dataset...')
            self.data = self.pretransform_data(self.files)
            self.labels = self.labels
        self.data = self.data.to(self.datasetdevice)
        self.labels = self.labels.to(self.datasetdevice)
        
    def download(self):
        """Download the CelebA dataset if it doesn't exist in self.root."""
        if self._check_exists():
            return
        if self._pretransformed_exists():
            self.faster_load = True
            return

        logging.warning("""Unfortunately downloading CelebA dataset is problematic due to Google Drive download quotas.
It is also an issue for even PyTorch. To use the dataset please download the dataset from Kaggle:
https://www.kaggle.com/datasets/jessicali9530/celeba-dataset/data 
Place the celeba.zip file in /data/CelebA """)

        zip_path = os.path.join(self.root, DATA_ZIP_FILE)

        # Verify the zip file
        if not zipfile.is_zipfile(zip_path):
            raise RuntimeError('Uploaded file is not a valid zip file.')

        # Extract the downloaded zip file
        print('Extracting CelebA dataset...')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.root)

        print('Download and extraction completed.')

    def _check_exists(self):
        """Check if the dataset exists."""
        required_files = [
            os.path.join(self.root, "list_attr_celeba.csv"),
            os.path.join(self.root, "list_eval_partition.csv"),
            os.path.join(self.root, "img_align_celeba")
        ]
        return all(os.path.exists(f) for f in required_files)

    def _pretransformed_exists(self):
        pretransformed_data_file = os.path.join(self.root, f'pretransformed_{self.split}.pt')
        pretransformed_labels_file = os.path.join(self.root, f'pretransformed_labels_{self.split}.pt')
        return os.path.exists(pretransformed_data_file) and os.path.exists(pretransformed_labels_file)

    def pretransform_data(self, file_paths):
        """Apply transformations to the dataset."""
        transformed_data = []
        for img_path in file_paths:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            transformed_data.append(img)
        return torch.stack(transformed_data)

    def __getitem__(self, index):
        if self.pre_transform:
            img = self.data[index]
        else:
            img_path = self.files[index].rstrip()
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
        
        labels = self.labels[index]
        labels_unique = self.labels_unique[index]
        return [img] + [label for label in labels] + [labels_unique]

    def __len__(self):
        return len(self.labels)

    def generate_unique_labels(self, labels):
        """Generate unique labels based on all task labels."""
        unique_labels = []
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
        for label_set in labels:
            unique_label = int(''.join(map(str, label_set)), 2)
            unique_labels.append(unique_label)
        return torch.tensor(unique_labels)

    def split_labels(self):
        # Predefined deterministic groups based on the attribute names you provided
        grouped_indices = [
            [0, 9, 3, 15, 22, 33, 25, 29],  # Group 1
            [1, 8, 7, 10, 16, 32, 26, 34],  # Group 2
            [2, 11, 12, 14, 21, 28, 35, 39],  # Group 3
            [4, 6, 13, 18, 19, 20, 30, 31],  # Group 4
            [5, 17, 23, 24, 27, 36, 37, 38]  # Group 5
        ]
        
        # Reshape the labels into 5 groups with 8 binary classifications each
        new_labels = torch.zeros((self.labels.shape[0], 5, 8), dtype=torch.long, device = self.labels.device)
        
        for i, group in enumerate(grouped_indices):
            # Select corresponding 8 labels for each group based on the deterministic grouping
            new_labels[:, i, :] = self.labels[:, group]
        
        self.labels = new_labels


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
    for attr in ['root', 'transform', 'train', 'multi', 'datasetdevice', 'pre_transform']:
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
import torch.nn.init as init
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset
import math
import torchvision
import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.models as models

# Implementation below is taken from FedExp https://github.com/Divyansh03/FedExP/blob/main/util_models.py
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(2,planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(2,planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(2,self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(2,64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 3)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
class ResNet_Backbone(nn.Module): 
    def __init__(self):
        super(ResNet_Backbone,self).__init__()
        
        # Use the resnet18 function defined previously to create a ResNet model with BasicBlock
        self.model = ResNet(BasicBlock, [2, 2, 2, 2])
        
        # Remove the final fully connected layer
        self.model.linear = nn.Identity()
        
    def forward(self, x, mask):
        # Forward through the modified ResNet (without the final linear layer)
        out = self.model(x)
        return out, mask

class FaceAttributeDecoder(nn.Module):
    def __init__(self):
        super(FaceAttributeDecoder, self).__init__()
        self.linear = nn.Linear(512, 8)  # Output length-8 vector

    def forward(self, x, mask):
        x = self.linear(x)
        out = torch.sigmoid(x)  # Apply sigmoid to each entry for probability
        return out, mask

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_model(config) -> object:
    device = config['model_device']
    model = {}
    model['rep'] = ResNet_Backbone()
    model['rep'].to(device)
    for t in get_tasks():
        model[t] = FaceAttributeDecoder()
        model[t].to(device)
    return model

import torch.nn.functional as F

def bce_loss(output, target):
    # Convert target (labels) to float
    target = target.float()
    
    # Compute Binary Cross-Entropy loss
    return F.binary_cross_entropy(output, target, reduction='mean')

def get_loss():
    loss_fn = {}
    for task in get_tasks():
        # Assign the custom BCE loss function for each task
        loss_fn[task] = bce_loss
    return loss_fn

def get_tasks():
    return ['T'+str(i) for i in range(1,6)]


def get_tasks_names():
    all_task_names = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs',
        'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',      
        'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',       
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 
        'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 
        'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
    ]
    
    # Grouped indices based on the provided grouping strategy
    grouped_indices = [
        [0, 9, 3, 15, 22, 33, 25, 29],  # Group 1
        [1, 8, 7, 10, 16, 32, 26, 34],  # Group 2
        [2, 11, 12, 14, 21, 28, 35, 39],  # Group 3
        [4, 6, 13, 18, 19, 20, 30, 31],  # Group 4
        [5, 17, 23, 24, 27, 36, 37, 38]  # Group 5
    ]
    
    # Join task names with a space separator for each group
    grouped_task_names = [' '.join([all_task_names[i] for i in group]) for group in grouped_indices]
    
    return grouped_task_names



if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt
    from torchvision import transforms

    # Instantiate the dataset
    dataset = Dataset(split='train', download=True, transform=global_transformer(), multi=True)

    # Create a DataLoader
    loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=10)

    # Loop through the DataLoader and visualize the data
    for dat in loader:
        ims = np.array((dat[0]*255+torch.tensor(MEAN).view(1,3,1,1)).permute(0, 2, 3, 1),dtype = np.uint8) # Convert to HWC for visualization
        f, axarr = plt.subplots(2, 5, figsize=(15, 6))
        for j in range(5):
            for i in range(2):
                t = ims[j * 2 + i]
                a = t
                axarr[i][j].imshow(a)
                axarr[i][j].axis('off')
    
        plt.show()
        a = input()
        if a == 'ex':
            break
        else:
            plt.close()

# The list of CelebA attributes
	# •	5_o_Clock_Shadow (Beard-related)
	# •	Blond_Hair (Hair-related)
	# •	Bags_Under_Eyes (Skin-related)
	# •	Eyeglasses (Accessory-related)
	# •	Mustache (Facial hair)
	# •	Wavy_Hair (Hair-related)
	# •	Oval_Face (Face shape)
	# •	Rosy_Cheeks (Skin-related)
	# 2.	Group 2:
	# •	Arched_Eyebrows (Eyebrows-related)
	# •	Black_Hair (Hair-related)
	# •	Big_Nose (Face shape)
	# •	Blurry (Image quality)
	# •	Goatee (Facial hair)
	# •	Straight_Hair (Hair-related)
	# •	Pale_Skin (Skin-related)
	# •	Wearing_Earrings (Accessory-related)
	# 3.	Group 3:
	# •	Attractive (General appearance)
	# •	Brown_Hair (Hair-related)
	# •	Bushy_Eyebrows (Eyebrows-related)
	# •	Double_Chin (Face shape)
	# •	Mouth_Slightly_Open (Expression)
	# •	Receding_Hairline (Hair-related)
	# •	Wearing_Hat (Accessory-related)
	# •	Young (Age-related)
	# 4.	Group 4:
	# •	Bald (Hair-related)
	# •	Big_Lips (Face shape)
	# •	Chubby (Body shape)
	# •	Heavy_Makeup (Makeup-related)
	# •	High_Cheekbones (Face shape)
	# •	Male (Gender)
	# •	Sideburns (Facial hair)
	# •	Smiling (Expression)
	# 5.	Group 5:
	# •	Bangs (Hair-related)
	# •	Gray_Hair (Hair-related)
	# •	Narrow_Eyes (Eye-related)
	# •	No_Beard (Facial hair)
	# •	Pointy_Nose (Face shape)
	# •	Wearing_Lipstick (Makeup-related)
	# •	Wearing_Necklace (Accessory-related)
	# •	Wearing_Necktie (Accessory-related)


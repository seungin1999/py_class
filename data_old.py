import os
import numpy as np
import torch
import torchvision as tv
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, Subset
from torchvision import datasets
import math

def get_log_num_of_samples(x, classes, max_data):
    log_den = classes/(10**((classes-1)/classes)-1)
    return int(max_data*(-math.log(x/log_den+1)+1))

def get_data_and_labels(dataset):
    if isinstance(dataset, Subset):
        data = dataset.dataset.data[dataset.indices]
        if hasattr(dataset.dataset, 'targets'):
            labels = np.array(dataset.dataset.targets)[dataset.indices]
        else:
            labels = np.array(dataset.dataset.labels)[dataset.indices]
    else:
        data = dataset.data
        labels = dataset.targets if hasattr(dataset, 'targets') else dataset.labels
    return data, label

def get_loader(data, data_path, batch_size, imbalanced_cifar = False):
    # dataset normalize values
    if data == 'cifar100':
        mean = [0.507, 0.487, 0.441]
        stdv = [0.267, 0.256, 0.276]
    elif data == 'cifar10':
        mean = [0.491, 0.482, 0.447]
        stdv = [0.247, 0.243, 0.262]
    elif data == 'svhn':
        mean = [0.5, 0.5, 0.5]
        stdv = [0.5, 0.5, 0.5]

    # augmentation
    train_transforms = tv.transforms.Compose([
        tv.transforms.RandomCrop(32, padding=4),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])

    test_transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])

    # load datasets
    if data == 'cifar100':
        train_set_full = datasets.CIFAR100(root=os.path.join(data_path, 'cifar100_data'),
                                      train=True,
                                      transform=train_transforms,
                                      download=True)
        test_set = datasets.CIFAR100(root=os.path.join(data_path, 'cifar100_data'),
                                     train=False,
                                     transform=test_transforms,
                                     download=False)
        
        if imbalanced_cifar:
            indices = [[] for _ in range(100)]

            classes = 100

            for i in range(len(train_set_full)):
                indices[train_set_full[i][1]].append(i)
            for i in range(100):
                indices[i] = indices[i][:get_log_num_of_samples(i, classes=classes, max_data=10000)]

            indices = list(np.array(indices[0]).flatten())

            train_set = Subset(train_set_full, indices)
        else:
            train_set = train_set_full

    elif data == 'cifar10':  # cifar10_data /cifiar10_data
        train_set_full = datasets.CIFAR10(root=os.path.join(data_path, 'cifar10_data'),
                                     train=True,
                                     transform=train_transforms,
                                     download=True)
        test_set = datasets.CIFAR10(root=os.path.join(data_path, 'cifar10_data'),
                                    train=False,
                                    transform=test_transforms,
                                    download=False)
        classes = 10

        if imbalanced_cifar:
            indices = [[] for _ in range(10)]

            for i in range(len(train_set_full)):
                indices[train_set_full[i][1]].append(i)
            for i in range(10):
                indices[i] = indices[i][:get_log_num_of_samples(i, classes=classes, max_data=10000)]

            indices = list(np.array(indices[0]).flatten())

            train_set = Subset(train_set_full, indices)
        else:
            train_set = train_set_full

    elif data == 'svhn':
        train_set = datasets.SVHN(root=os.path.join(data_path, 'svhn_data'),
                                  split='train',
                                  transform=train_transforms,
                                  download=True)
        test_set = datasets.SVHN(root=os.path.join(data_path, 'svhn_data'),
                                 split='test',
                                 transform=test_transforms,
                                 download=True)

    # make Custom_Dataset
    if data == 'svhn':
        train_data = Custom_Dataset(train_set.data,
                            
                                    train_set.labels,
                                    'svhn', train_transforms)
        test_data = Custom_Dataset(test_set.data,
                                   test_set.labels,
                                   'svhn', test_transforms)
        # one_hot_encoding
        test_onehot = one_hot_encoding(test_set.labels)
        test_label = test_set.labels
    else:
        # train_data_array, train_labels = get_data_and_labels(train_set)
        # test_data_array, test_labels = get_data_and_labels(test_set)
        
        # train_data = Custom_Dataset(train_data_array,
        #                         train_labels,
        #                         'cifar', train_transforms)
        # test_data = Custom_Dataset(test_data_array,
        #                        test_labels,
        #                        'cifar', test_transforms)
        
        train_data = Custom_Dataset(train_set.data,
                                    train_set.targets,
                                    'cifar', train_transforms)
        
        test_data = Custom_Dataset(test_set.data,
                                   test_set.targets,
                                   'cifar', test_transforms)
        # one_hot_encoding
        test_onehot = one_hot_encoding(test_set.targets)
        test_label = test_set.targets


    # make DataLoader
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4)

    print("-------------------Make loader-------------------")
    print('Train Dataset :',len(train_loader.dataset),
          '   Test Dataset :',len(test_loader.dataset))

    return train_loader, test_loader, test_onehot, test_label
# Custom_Dataset class
class Custom_Dataset(Dataset):
    def __init__(self, x, y, data_set, transform=None):
        self.x_data = x
        self.y_data = y
        self.data = data_set
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    # return idx
    def __getitem__(self, idx):
        if self.data == 'cifar':
            img = Image.fromarray(self.x_data[idx])
        elif self.data == 'svhn':
            img = Image.fromarray(np.transpose(self.x_data[idx], (1, 2, 0)))

        x = self.transform(img)

        return x, self.y_data[idx], idx

def one_hot_encoding(label):
    print("one_hot_encoding process")
    cls = set(label)
    class_dict = {c: np.identity(len(cls))[i, :] for i, c in enumerate(cls)}
    one_hot = np.array(list(map(class_dict.get, label)))

    return one_hot


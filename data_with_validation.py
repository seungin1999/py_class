# %%
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
    log_den = (classes-1)/(10**(99/100)-1)
    # print(x, int(max_data*(-math.log(x/log_den+1, 10)+1)))
    return int(max_data*(-math.log(x/log_den+1, 10)+1))

def get_loader(data, data_path, batch_size, imbalanced_cifar = False, ratio=0.2):
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
        train_set = datasets.CIFAR100(root=os.path.join(data_path, 'cifar100_data'),
                                      train=True,
                                      transform=train_transforms,
                                      download=True)
        test_set = datasets.CIFAR100(root=os.path.join(data_path, 'cifar100_data'),
                                     train=False,
                                     transform=test_transforms,
                                     download=False)
        
        # if imbalanced_cifar:
        #     indices = [[] for _ in range(100)]

        #     classes = 100

        #     for i in range(len(train_set_full)):
        #         indices[train_set_full[i][1]].append(i)
        #     for i in range(100):
        #         indices[i] = indices[i][:get_log_num_of_samples(i, classes=classes, max_data=2000)]

        #     indices = list(np.array(indices).flatten())

        #     train_set = Subset(train_set_full, indices)
        # else:
        #     train_set = train_set_full

    elif data == 'cifar10':  # cifar10_data /cifiar10_data
        train_set = datasets.CIFAR10(root=os.path.join(data_path, 'cifar10_data'),
                                     train=True,
                                     transform=train_transforms,
                                     download=True)
        test_set = datasets.CIFAR10(root=os.path.join(data_path, 'cifar10_data'),
                                    train=False,
                                    transform=test_transforms,
                                    download=False)
        # classes = 10

        # if imbalanced_cifar:
        #     indices = [[] for _ in range(10)]

        #     for i in range(len(train_set_full)):
        #         indices[train_set_full[i][1]].append(i)
        #     for i in range(100):
        #         indices[i] = indices[i][:get_log_num_of_samples(i, classes=classes, max_data=2000)]

        #     indices = list(np.array(indices).flatten())

        #     train_set = Subset(train_set_full, indices)
        # else:
        #     train_set = train_set_full

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
        # Code on creating validation dataset below
        train_set_data, train_set_targets, val_set_data, val_set_targets = train_validation_split(train_set.data, train_set.targets, ratio, data)
        #

        train_data = Custom_Dataset(train_set_data,
                                    train_set_targets,
                                    data, train_transforms,
                                    imbalanced=imbalanced_cifar)
        val_data = Custom_Dataset(val_set_data,
                                  val_set_targets,
                                  data, test_transforms,
                                  imbalanced=imbalanced_cifar)
        test_data = Custom_Dataset(test_set.data,
                                   test_set.targets,
                                   data, test_transforms,
                                   imbalanced=imbalanced_cifar)
        # one_hot_encoding
        test_onehot = one_hot_encoding(test_set.targets, 100 if data == "cifar100" else 10, imbalanced=imbalanced_cifar)

        classes = 10 if data=="cifar10" else 100

        #changed so that test_label fits actual test set
        new_y = []
        if imbalanced_cifar:
            for cls in range(classes):
                new_y += [cls] * get_log_num_of_samples(cls, classes, int(len(test_set.targets)/classes))
        else:
            new_y = test_set.targets
        test_label = new_y


    # make DataLoader
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4)

    print("-------------------Make loader-------------------")
    print('Train Dataset :',len(train_loader.dataset),
          '   Validation Dataset :', len(val_loader.dataset),
          '   Test Dataset :',len(test_loader.dataset))

    return train_loader, test_loader, test_onehot, test_label, val_loader
# Custom_Dataset class
class Custom_Dataset(Dataset):
    def __init__(self, x, y, data_set, transform=None, test=False, imbalanced=False):
        gt_len = 0
        new_x = []
        classes = 0
        if imbalanced and not test:
            if data_set == "cifar10":
                new_x = [[] for _ in range(10)]
                classes = 10
                # print(f"Length of new_x: {len(new_x)}")
            elif data_set == "cifar100":
                new_x = [[] for _ in range(100)]
                classes = 100
                # print(f"Length of new_x: {len(new_x)}")

            else:
                print("Error in Custom Dataset")

            new_y = []

            # print(f"\n\nLength of y: {len(y)}")

            for i in range(len(y)):
                new_x[y[i]].append(x[i])
                # print(f"value of y in index {i}: {y[i]}")

            # for i in new_x:
            #     print(f"Length of each data: {len(i)}")

            max_data = len(new_x[0])

            for cls in range(len(new_x)):
                gt_len += get_log_num_of_samples(cls, classes, max_data)
                new_x[cls] = new_x[cls][:get_log_num_of_samples(cls, classes, max_data)]
                new_y += [cls] * get_log_num_of_samples(cls, classes, max_data)
            
            self.x_data = np.array([item for row in new_x for item in row])
            self.y_data = np.array(new_y)
        
        else:
            self.x_data = x
            self.y_data = y
            
        if 'cifar' in data_set:
            self.data = 'cifar'
        else:
            self.data = data_set
        self.transform = transform

        # print("\n\n", len(self.x_data), len(self.y_data), gt_len)

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

def one_hot_encoding(label, classes=10, imbalanced=False):
    print("one_hot_encoding process")

    new_y = []
    if imbalanced:
        for cls in range(classes):
            new_y += [cls] * get_log_num_of_samples(cls, classes, int(len(label)/classes))
    else:
        new_y = label
    cls = set(new_y)
    class_dict = {c: np.identity(len(cls))[i, :] for i, c in enumerate(cls)}
    one_hot = np.array(list(map(class_dict.get, new_y)))

    return one_hot

def train_validation_split(data, targets, ratio, dataset):
    train_data, train_targets, val_data, val_targets = [], [], [], []
    data_count = {}
    classes = 100 if dataset == "cifar100" else 10
    for index in range(len(targets)):
        # Data per class counter
        if targets[index] in data_count.keys():
            data_count[targets[index]] += 1
        else:
            data_count[targets[index]] = 1

        # appending to train data and val data based on how manay data has been added so far to train data
        if data_count[targets[index]] <=  int(len(data)*(1-ratio)/classes):
            train_data.append(data[index])
            train_targets.append(targets[index])
        else:
            val_data.append(data[index])
            val_targets.append(targets[index])

    # Convert the data into numpy arrays (as was the original)
    train_data = np.array(train_data)
    val_data = np.array(val_data)

    # print(len(list(data)), type(targets))

    return train_data, train_targets, val_data, val_targets
# %%

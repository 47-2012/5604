import os
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def get_dataloader(path, mode='training', augment=True, batch_size=1, shuffle=True, num_workers=1, pin_memory=True, val_split=0, flow=False, drop_last=False):
    if flow:
        dataset = Vimeo90KFlow(path, mode, augment=augment, val_split=val_split)
    else:
        dataset = Vimeo90K(path, mode, augment=augment, val_split=val_split)
    if mode == 'training':
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last), dataset.val_samples
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)


class Vimeo90K(Dataset):
    def __init__(self, path, mode, augment=True, split_ratio=0.9, val_split=0):
        self.mode = mode
        self.augment = augment
        self.images_path = os.path.join(path, 'sequences')

        # Create list for the samples
        train_path = os.path.join(path, 'tri_trainlist.txt')
        #test_path = os.path.join(path, 'tri_testlist.txt')
        test_path = os.path.join(path, 'sub1_tri_testlist.txt')
        with open(train_path, 'r') as f:
            train_samples = np.asarray(f.read().splitlines())
            if mode == 'training':
                samples = np.random.permutation(len(train_samples))
                ratio = int(split_ratio * len(train_samples))
                self.train_samples = train_samples[samples[:ratio]]
                self.val_samples = samples[ratio:]
            elif mode == 'validation':
                self.val_samples = train_samples[val_split]
        with open(test_path, 'r') as f:
            self.test_samples = f.read().splitlines()
        
        # Set augmentation operations
        if self.augment:
            self.transform = transforms.Compose([
                transforms.RandomCrop(size=(256,256)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                # Add ColorJitter? Is not performed uniformly across the frames? Or it is with torch.random_seed
                # Maybe https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py#L434 
                #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
            ])
        
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        # List image paths
        if self.mode == 'training':
            img_path = os.path.join(self.images_path, self.train_samples[index])
        elif self.mode == 'validation':
            img_path = os.path.join(self.images_path, self.val_samples[index])
        else:
            img_path = os.path.join(self.images_path, self.test_samples[index])
        
        img_paths = [img_path + '/im1.png', img_path + '/im2.png', img_path + '/im3.png']


        # Load images
        img1 = Image.open(img_path+'/im1.png')
        img2 = Image.open(img_path+'/im2.png')
        img3 = Image.open(img_path+'/im3.png')

        
        # Perform augmentation
        if self.mode == 'training' and self.augment:
            # Each image different transform? So add random seed between each image?
            seed = random.randint(0, 2**32)
            set_seed(seed)
            img1 = self.transform(img1)
            set_seed(seed)
            img2 = self.transform(img2)
            set_seed(seed)
            img3 = self.transform(img3)

            
            # Random temporal flip
            if random.random() > 0.5:
                img1, img3 = img3, img1
                img_paths[0], img_paths[2] = img_paths[2], img_paths[0]

        # Transform PIL to Tensor
        img1 = self.to_tensor(img1)
        img2 = self.to_tensor(img2)
        img3 = self.to_tensor(img3)
        
        return (img1, img2, img3), img_paths

    def __len__(self):
        # Return length of training or test samples
        if self.mode == 'training':
            return len(self.train_samples)
        elif self.mode =='validation':
            return len(self.val_samples)
        else:
            return len(self.test_samples)


class Vimeo90KFlow(Dataset):
    def __init__(self, path, mode, augment=True, split_ratio=0.9, val_split=0):
        self.mode = mode
        self.augment = augment
        self.images_path = os.path.join(path, 'sequences')

        # Create list for the samples
        train_path = os.path.join(path, 'tri_trainlist.txt')
        test_path = os.path.join(path, 'tri_testlist.txt')

        with open(train_path, 'r') as f:
            train_samples = np.asarray(f.read().splitlines())
            if mode == 'training':
                samples = np.random.permutation(len(train_samples))
                ratio = int(split_ratio * len(train_samples))
                self.train_samples = train_samples[samples[:ratio]]
                self.val_samples = samples[ratio:]
            elif mode == 'validation':
                self.val_samples = train_samples[val_split]
        with open(test_path, 'r') as f:
            self.test_samples = f.read().splitlines()
        
        # Set augmentation operations
        if self.augment:
            self.transform = transforms.Compose([
                transforms.RandomCrop(size=(256,256)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                # Add ColorJitter? Is not performed uniformly across the frames? Or it is with torch.random_seed
                # Maybe https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py#L434 
                #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
            ])
        
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        # List image paths
        if self.mode == 'training':
            img_path = os.path.join(self.images_path, self.train_samples[index])
        elif self.mode == 'validation':
            img_path = os.path.join(self.images_path, self.val_samples[index])
        else:
            img_path = os.path.join(self.images_path, self.test_samples[index])
        
        img_paths = [img_path + '/im1.png', img_path + '/im2.png', img_path + '/im3.png']


        # Load images
        img1 = Image.open(img_path+'/im1.png')
        img2 = Image.open(img_path+'/im2.png')
        img3 = Image.open(img_path+'/im3.png')
        ft_0 = torch.from_numpy(np.copy(read_flow(img_path+'/im2_to_im1.flo')))
        ft_1 = torch.from_numpy(np.copy(read_flow(img_path+'/im2_to_im3.flo')))


        
        # Perform augmentation
        if self.mode == 'training' and self.augment:
            # Each image different transform? So add random seed between each image?
            seed = random.randint(0, 2**32)
            set_seed(seed)
            img1 = self.transform(img1)
            set_seed(seed)
            img2 = self.transform(img2)
            set_seed(seed)
            img3 = self.transform(img3)
            set_seed(seed)
            ft_0 = self.transform(ft_0)
            set_seed(seed)
            ft_1 = self.transform(ft_1)

            
            # Random temporal flip
            if random.random() > 0.5:
                img1, img3 = img3, img1
                ft_0, ft_1 = ft_1, ft_0
                img_paths[0], img_paths[2] = img_paths[2], img_paths[0]

        # Transform PIL to Tensor
        img1 = self.to_tensor(img1)
        img2 = self.to_tensor(img2)
        img3 = self.to_tensor(img3)
        
        return (img1, img2, img3, ft_0, ft_1), img_paths

    def __len__(self):
        # Return length of training or test samples
        if self.mode == 'training':
            return len(self.train_samples)
        elif self.mode =='validation':
            return len(self.val_samples)
        else:
            return len(self.test_samples)

def read_flow(flow_file):
    with open(flow_file, 'rb') as f:
        flow_f = f.read()
    data = np.frombuffer(buffer=flow_f, dtype=np.float32)
    assert(data[0] == 202021.25)
    w, h = np.frombuffer(buffer=flow_f, dtype=np.int32, count=2, offset=4)
    return data[3:].reshape((2, h, w))


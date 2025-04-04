import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, Subset
from imagenet_process import get_or_create_selected_classes, split_dataset, random_split_clients, MultiAugmentDataset
from datasets import load_dataset
import numpy as np
import csv, shutil
import json
import os
from collections import defaultdict

dataset_stats = {
    'mnist': {'mean': [0.1307], 'std': [0.3081]},
    'cifar10': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.24703233, 0.24348505, 0.26158768]},
    'cifar100': {'mean': [0.5071, 0.4867, 0.4408], 'std': [0.2675, 0.2565, 0.2761]}
}

dataset_cfg = {
    'mnist': {'input_size': 28, 'output_size': 10, 'input_channel': 1},    
    'femnist': {'input_size': 28, 'output_size': 62, 'input_channel': 1},    
    'cifar10': {'input_size': 32, 'output_size': 10, 'input_channel': 3},
    'imagenet': {'input_size': 64, 'output_size': 50, 'input_channel': 3},

}

def scale_crop(input_size, scale_size, normalize):
    t_list = [
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    if scale_size != input_size:
        t_list = [transforms.Resize(scale_size)] + t_list

    return transforms.Compose(t_list)

def pad_random_crop(input_size, scale_size, normalize):
    padding = int((scale_size - input_size) / 2)
    return transforms.Compose([
        transforms.RandomCrop(input_size, padding=padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ])
    
def iid_samples(dataset, num_users):
    total_indices = np.arange(len(dataset))
    np.random.shuffle(total_indices)

    ds = []
    num_data = len(dataset) // num_users
    for i in range(num_users):
        indices = total_indices[i*num_data:(i+1)*num_data]
        ds.append(Subset(dataset, indices))
    return ds

def dirichlet_sample(dataset, num_clients, num_classes, alpha=0.5):
    labels = np.array(dataset.dataset.targets)[dataset.indices.astype(int)]
    
    # Create a list of indices for each class
    indices_per_class = [[] for _ in range(num_classes)]
    for idx, label in enumerate(labels):
        indices_per_class[label].append(idx)
    
    # Create Dirichlet distribution for the non-IID partitioning
    dirichlet_dist = np.random.dirichlet([alpha] * num_clients, num_classes)
    
    client_indices = [[] for _ in range(num_clients)]
    for cls, proportions in zip(indices_per_class, dirichlet_dist):
        cls_indices = np.array(cls)
        np.random.shuffle(cls_indices)
        
        # Split the class indices according to the proportions
        split_indices = np.split(cls_indices, (np.cumsum(proportions)[:-1] * len(cls)).astype(int))
        for client_idx, indices in enumerate(split_indices):
            client_indices[client_idx].extend(indices)
    return  [Subset(dataset, indices) for indices in client_indices]
    
def get_fl_dataset(args, num_data_per_client, num_clients):
    if 'mnist' == args.dataset:
        train_ds_clients, test_ds_clients, test_ds  = get_mnist_data(args, num_data_per_client, num_clients)
    elif 'femnist' == args.dataset:
        train_ds_clients, test_ds_clients, test_ds  = get_femnist_data(args)
    elif 'cifar10' == args.dataset:
        train_ds_clients, test_ds_clients, test_ds  = get_cifar10_data(args, num_data_per_client, num_clients)
    elif 'imagenet' == args.dataset:
        train_ds_clients, test_ds_clients, test_ds  = get_imagenet_data(args, num_data_per_client, num_clients)
    return train_ds_clients, test_ds_clients, test_ds 


def get_cifar10_data(args, num_data_per_client, num_clients):
    norm = dataset_stats['cifar10']
    train_ds = datasets.CIFAR10('data/cifar10', train=True, download=True, 
                                     transform=pad_random_crop(input_size=32,
                                                            scale_size=40, normalize=norm),)
    test_ds = datasets.CIFAR10('data/cifar10', train=False, download=True, 
                                    transform=scale_crop(input_size=32,
                                                        scale_size=32, normalize=norm),)
    number_data = num_data_per_client * num_clients
    print(len(train_ds), number_data, len(test_ds))
    train_ds = Subset(train_ds, np.random.choice(len(train_ds), number_data, replace=False))
    test_ds = Subset(test_ds, np.random.choice(len(test_ds), min(int(number_data/2), len(test_ds)), replace=False))
    if args.niid:
        train_ds_clients = dirichlet_sample(train_ds, num_clients, 10)
        test_ds_clients = dirichlet_sample(test_ds, num_clients, 10)
    else:
        train_ds_clients = iid_samples(train_ds, num_clients)
        test_ds_clients = iid_samples(test_ds, num_clients)
    return train_ds_clients, test_ds_clients, test_ds

def get_mnist_data(args, num_data_per_client, num_clients):
    cfg = dataset_cfg[args.dataset]
    transform = transforms.Compose([
        transforms.Resize(cfg['input_size']),
        transforms.ToTensor(),
    ])
    norm = dataset_stats['mnist']
    train_ds = datasets.MNIST('data/mnist', train=True, download=True, 
                                    transform=scale_crop(input_size=28,
                                                    scale_size=cfg['input_size'], normalize=norm))
    test_ds = datasets.MNIST('data/mnist', train=False, download=True, 
                                transform=scale_crop(input_size=28,
                                                    scale_size=cfg['input_size'], normalize=norm))
    number_data = num_data_per_client * num_clients
    train_ds = Subset(train_ds, np.random.choice(len(train_ds), number_data, replace=False))
    test_ds = Subset(test_ds, np.random.choice(len(test_ds), min(int(number_data/2), len(test_ds)), replace=False))
    if args.niid:
        train_ds_clients = dirichlet_sample(train_ds, num_clients, 10)
        test_ds_clients = dirichlet_sample(test_ds, num_clients, 10)
    else:
        train_ds_clients = iid_samples(train_ds, num_clients)
        test_ds_clients = iid_samples(test_ds, num_clients)
    return train_ds_clients, test_ds_clients, test_ds

def get_imagenet_data(args, num_data_per_client, num_clients):
    dataset = load_dataset("benjamin-paine/imagenet-1k-64x64")
    train_subset, val_subset, selected_classes, label_map = get_or_create_selected_classes(dataset, save_path="selected_classes.json", num_selected=50)
    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),    # Random crop with padding.
        transforms.RandomHorizontalFlip(),        # Random horizontal flip.
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    def transform_train(example):
    # Applies training augmentation to the image.
        example["image"] = train_transform(example["image"])
        return example

    def transform_val(example):
        # Applies validation transform to the image.
        example["image"] = val_transform(example["image"])
        return example

    # train_ds = train_subset.with_transform(transform_train)
    # test_ds = val_subset.with_transform(transform_val)
    # train_ds_clients = split_dataset(train_ds, num_clients, num_data_per_client)
    # test_ds_clients = split_dataset(test_ds, num_clients)
    train_ds = HFDatasetWrapper(train_subset, transform=train_transform)
    test_ds = HFDatasetWrapper(val_subset, transform=val_transform)
    train_ds = MultiAugmentDataset(train_ds, n_views=3)
    train_ds_clients = random_split_clients(train_ds, num_clients, num_data_per_client)
    test_ds_clients = random_split_clients(test_ds, num_clients, len(test_ds)//num_clients)
    return train_ds_clients, test_ds_clients, test_ds



    
def read_dir(data_dir):
    clients = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, data


def read_data(train_data_dir, test_data_dir):
    train_clients, train_data = read_dir(train_data_dir)
    test_clients, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    return train_clients, train_data, test_data


def get_femnist_data(args):
    if args.niid:
        train_clients, train_data, test_data = read_data('data/femnist_niid_140/train', 'data/femnist_niid_140/test') 
    else:
        train_clients, train_data, test_data = read_data('data/femnist_iid_36/train', 'data/femnist_iid_36/test') 

    train_ds_clients = []
    test_ds_clients = []
    ttx = []
    tty = []

    for c in train_clients:
        train_ds_clients.append(DatasetFEMNIST(train_data[c]['x'], train_data[c]['y']))
        tx = test_data[c]['x']
        ty = test_data[c]['y']
        test_ds_clients.append(DatasetFEMNIST(tx, ty))
        ttx.extend(tx)
        tty.extend(ty)
    return train_ds_clients, test_ds_clients, DatasetFEMNIST(ttx, tty)

class Dataset_Custom(Dataset):
    def __init__(self, data, hist_len):
        # init
        self.data = data
        self.hist_len = hist_len

    def __getitem__(self, index):
        if self.hist_len == 1:
            return self.data[index], self.data[index]
        return self.data[index*self.hist_len:index*self.hist_len+self.hist_len], self.data[index*self.hist_len:index*self.hist_len+self.hist_len]
    
    def __len__(self):
        return len(self.data)//self.hist_len

class DatasetFEMNIST(Dataset):
    def __init__(self, data, label):
        # init
        self.data = torch.stack([torch.tensor(d) for d in data]).reshape(-1,28, 28)
        self.label = torch.tensor(label)
        mean = self.data.mean()
        std = self.data.std()
        self.data = (self.data - mean) / std

    def __getitem__(self, index):
        return self.data[index].unsqueeze(0), int(self.label[index])
    
    def __len__(self):
        return len(self.data)
    

class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, transform=None):
        """
        Args:
            hf_dataset: A Hugging Face Dataset object.
            transform: A callable transformation to be applied on a sample.
        """
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __getitem__(self, index):
        # Access the sample (decoded on-demand)
        sample = self.hf_dataset[index]
        image, label = sample["image"], sample["label"]

        # Optionally apply transformation to the image
        if self.transform:
            image = self.transform(image)
        
        return image, label

    def __len__(self):
        return len(self.hf_dataset)


# def get_imagenet_tiny_data(args, num_data_per_client, num_clients):
#     data_dir = 'data/tiny-imagenet-200'
#     if not os.path.exists(data_dir):
#         raise FileNotFoundError(f"Directory {data_dir} does not exist.")


#     train_transform = transforms.Compose([
#         transforms.Resize((64, 64)),
#         transforms.RandomCrop(64, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(10),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.480, 0.448, 0.398], std=[0.277, 0.269, 0.282]),
#     ])

#     # Use a simpler transform for validation (no randomness)
#     val_transform = transforms.Compose([
#         transforms.Resize((64, 64)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.480, 0.448, 0.398], std=[0.277, 0.269, 0.282]),
#     ])


# # Datasets
#     train_dir = data_dir + '/train'
#     val_dir = data_dir + '/val'
#     class_file = 'selected_classes.json'

#     # Get selected classes
#     selected_classes = get_or_create_selected_classes(train_dir, class_file)

#     train_ds = SubclassFilter(train_dir, selected_classes, train_transform)
#     test_ds = SubclassFilter(val_dir, selected_classes, val_transform)   
#     train_ds = MultiAugmentDataset(train_ds, n_views=3)  # For example, 3 augmented views per image


#     if args.niid:
#         train_ds_clients = dirichlet_sample(train_ds, num_clients, 50)
#         test_ds_clients = dirichlet_sample(test_ds, num_clients, 50)
#     else:
#         train_ds_clients = random_split_clients(train_ds, num_clients, num_data_per_client)
#         test_ds_clients = random_split_clients(test_ds, num_clients, len(test_ds)//num_clients)

#     return train_ds_clients, test_ds_clients, test_ds
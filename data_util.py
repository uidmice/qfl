import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import csv

CWRU_hist = 32
dataset_stats = {
    'mnist': {'mean': [0.1307], 'std': [0.3081]},
    'cifar10': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.24703233, 0.24348505, 0.26158768]},
    'cifar100': {'mean': [0.5071, 0.4867, 0.4408], 'std': [0.2675, 0.2565, 0.2761]}
}

dataset_cfg = {
    'mnist': {'input_size': 28, 'output_size': 10, 'input_channel': 1},    
    'mnist_b': {'input_size': 28, 'output_size': 2, 'input_channel': 1},
    'mnist0': {'input_size': 14, 'output_size': 10, 'input_channel': 1},    
    'mnist0_b': {'input_size': 14, 'output_size': 2, 'input_channel': 1},
    'cifar10': {'input_size': 32, 'output_size': 10, 'input_channel': 3},
    'spectrum': {'input_size': 0, 'output_size': 64, 'input_channel': 64},
    'CWRU': {'input_size': 0, 'output_size': CWRU_hist, 'input_channel': CWRU_hist}
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

def get_dataset(args):
    if 'mnist' in args.dataset:
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

        if 'binary' in args.dataset:
            train_ds= [(x,label % 2) for (x,label) in train_ds]
            test_ds= [(x,label % 2) for (x,label) in test_ds]

    if 'cifar' in args.dataset:
        norm = dataset_stats['cifar10']
        
        train_ds = datasets.CIFAR10('data/cifar10', train=True, download=True, 
                                     transform=pad_random_crop(input_size=32,
                                                            scale_size=40, normalize=norm),)
        test_ds = datasets.CIFAR10('data/cifar10', train=False, download=True, 
                                    transform=scale_crop(input_size=32,
                                                        scale_size=32, normalize=norm),)
    if 'spectrum' in args.dataset:
        data = np.fromfile('data/Signal_data/USRP_B210_751MHZ_10MS', dtype=np.complex64)
        I = torch.from_numpy(np.real(data)) * 10
        Q = torch.from_numpy(np.imag(data)) * 10
        A = torch.sqrt(I**2 + Q**2)
        P = torch.atan2(Q, I)
        I = 2 * (I - I.min()) / (I.max() - I.min()) - 1
        Q = 2 * (Q - Q.min()) / (Q.max() - Q.min()) - 1
        A = 2 * (A - A.min()) / (A.max() - A.min()) - 1
        P = 2 * (P - P.min()) / (P.max() - P.min()) - 1
        data = torch.stack([I, Q, A, P], dim=1).flatten()
        test_data = int(len(data) * 0.15)
        # test_data = 10000
        train_data = len(data) - test_data
        train_ds = Dataset_Custom(data[:train_data], 64)
        test_ds = Dataset_Custom(data[train_data:], 64)

    if 'CWRU' in args.dataset:
        a = csv.reader(open('data/motor/X097_FE_time.txt', 'r'), delimiter=',')
        data = []
        for row in a:
            data += [float(row[0])]
        data = torch.tensor(data)
        data = 2 * (data - data.min()) / (data.max() - data.min()) - 1
        train_data = int(len(data) * 0.8)
        train_ds = Dataset_Custom(data[:train_data], CWRU_hist)
        test_ds = Dataset_Custom(data[train_data:], CWRU_hist)
    return train_ds, test_ds

def iid_samples(dataset, num_users):
    total_indices = np.arange(len(dataset))
    np.random.shuffle(total_indices)

    ds = []
    num_data = len(dataset) // num_users
    for i in range(num_users):
        indices = total_indices[i*num_data:(i+1)*num_data]
        ds.append(Subset(dataset, indices))
    return ds

def get_fl_dataset_iid(args, num_data_per_client, num_clients):
    if 'mnist' in args.dataset:
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
    test_ds = Subset(test_ds, np.random.choice(len(test_ds), min(number_data*2, len(test_ds)), replace=False))
    train_ds_clients = iid_samples(train_ds, num_clients)
    test_ds_clients = iid_samples(test_ds, num_clients)
    return train_ds_clients, test_ds_clients, test_ds 
def get_fl_dataset_niid(args, num_data_per_client, num_clients):
    if 'mnist' in args.dataset:
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
    test_ds = Subset(test_ds, np.random.choice(len(test_ds), min(number_data*2, len(test_ds)), replace=False))
    train_ds_clients = iid_samples(train_ds, num_clients)
    test_ds_clients = iid_samples(test_ds, num_clients)
    return train_ds_clients, test_ds_clients, test_ds 

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



    
# def prepare_mnist_data(dim, iid=0, num_dev = 5, num_ds = 300):

#     transform = transforms.Compose([
#         transforms.Resize((dim, dim)),  # Resize the image to 14x14
#         transforms.ToTensor(),
#         transforms.Lambda(lambda x: x.view(-1))
#     ])
#     mnist_train = datasets.MNIST('data/mnist', train=True, download=True, transform=transform)
#     mnist_test = datasets.MNIST('data/mnist', train=False, download=True, transform=transform)

#     mnist_even_odd_test= [(x,label % 2, label) for (x,label) in mnist_test]
#     mnist_even_odd_train= [(x,label % 2, label) for (x,label) in mnist_train]

#     if iid:
#         return mnist_even_odd_test, mnist_even_odd_train, *iid_samples(mnist_train, num_dev, num_ds)
#     return mnist_even_odd_test, mnist_even_odd_train, *mnist_non_iid_samples(mnist_train, num_dev, num_ds)

# def samples(dataset, num_users):
#     total_indices = np.arange(len(dataset))
#     np.random.shuffle(total_indices)

#     ds = []
#     num_data = len(dataset) // num_users
#     for i in range(num_users):
#         indices = total_indices[i*num_data:(i+1)*num_data]
#         original = [dataset[i] for i in indices]
#         ds.append([(x,label) for (x,label) in original])
#     return ds

# def prepare_thermal_data(num_dev = 5, num_ds = 300):

#     train_dataset = torch.load('data/thermal/train_dataset_scaled.pt')
#     test_dataset = torch.load('data/thermal/test_dataset_scaled.pt')

#     return test_dataset, train_dataset, samples(train_dataset, num_dev)

# def gen_loaders(path, BATCH_SIZE, NUM_WORKERS):
#     # Data loading code
#     train_dataset = datasets.CIFAR10(root=path,
#                                     train=True,
#                                     transform=pad_random_crop(input_size=32,
#                                                             scale_size=40),
#                                     download=True)

#     val_dataset = datasets.CIFAR10(root=path,
#                                     train=False,
#                                     transform=scale_crop(input_size=32,
#                                                         scale_size=32),
#                                     download=True)

#     train_loader = torch.utils.data.DataLoader(train_dataset,
#                                             batch_size=BATCH_SIZE,
#                                             shuffle=True,
#                                             num_workers=NUM_WORKERS,
#                                             pin_memory=True)

#     val_loader = torch.utils.data.DataLoader(val_dataset,
#                                             batch_size=BATCH_SIZE,
#                                             shuffle=False,
#                                             num_workers=NUM_WORKERS,
#                                             pin_memory=True)

#     return (train_loader, val_loader)

# def scale_crop(input_size, scale_size, normalize):
#     t_list = [
#         transforms.CenterCrop(input_size),
#         transforms.ToTensor(),
#         # transforms.Normalize(**normalize),
#     ]
#     if scale_size != input_size:
#         t_list = [transforms.Resize(scale_size)] + t_list

#     return transforms.Compose(t_list)

# def pad_random_crop(input_size, scale_size, normalize):
#     padding = int((scale_size - input_size) / 2)
#     return transforms.Compose([
#         transforms.RandomCrop(input_size, padding=padding),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(**normalize),
#     ])
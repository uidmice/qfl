from torch.utils.data import RandomSampler
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class Client:
    def __init__(self, client_id, train_data, eval_data, criterion, model_path, bitwidth_limit, group=None):
        self.id = client_id
        self.group = group
        self.train_data = train_data
        self.eval_data = eval_data
        self.criterion = criterion
        self.init_model = None
        self.model_path = model_path + f'/client_{client_id}.pth'
        self.bitwidth_limit = bitwidth_limit
        self.model_BW_hist = []
        self.train_COMM_hist = []
        self.train_epoch = 0

    def train(self, model, num_epochs=1, batch_size=32, num_sample=None, num_workers=4):
        if num_sample is None:
            num_sample = len(self.train_data)
        random_sample = RandomSampler(self.train_data, num_samples=num_sample)
        train_loader = DataLoader(self.train_data, batch_size=batch_size, sampler=random_sample, num_workers=num_workers)
        self.train_epoch += num_epochs
        for i in range(num_epochs):
            loss, acc = model.epoch(train_loader, i,  5, self.criterion, train=True)
        return loss, acc, model

    def test(self, model, set_to_use='test'):
        assert set_to_use in ['train', 'test', 'val']
        if set_to_use == 'train':
            data = self.train_data
        elif set_to_use == 'test' or set_to_use == 'val':
            data = self.eval_data
        test_loader = DataLoader(data, batch_size=128, shuffle=False)
        loss, acc = self.model.epoch(test_loader, 0,  -1, self.criterion, train=False)
        return loss, acc, model
    
class Server:
    def __init__(self, model):
        self.model = model
        self.selected_clients = []
        self.updates = []

    def select_clients_random(self, my_round, possible_clients, num_clients=5, prob=None):
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(my_round)
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False, p=prob)
        return [(len(c.train_data), len(c.eval_data)) for c in self.selected_clients]
    
    def update_client_model(self, clients):
        for c in clients:
            torch.save(self.model.state_dict(), c.model_path)
        return [c.id for c in clients]

    def get_clients_info(self, clients):
        if clients is None:
            clients = self.selected_clients

        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_samples = {c.id: len(c.train_data)+ len(c.eval_data) for c in clients}
        return ids, groups, num_samples
    
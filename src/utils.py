import torch
import random, os, shutil, logging

from torch.utils.tensorboard import SummaryWriter

import  numpy as np
from src.model import nn_fp, nn_q
from src.qm import QCELoss

class stat_record:
    def __init__(self):
        self.reset()

    def reset(self):
        self.data = []
        self.num = 0

    def update(self, val):
        self.data.append(val)
        self.num += 1

class layer_stats:
    def __init__(self):
        self.mean_w = stat_record()
        self.mean_g = stat_record()
        self.max_w = stat_record()
        self.max_g = stat_record()
        self.range_w = stat_record()
        self.range_g = stat_record()
        self.Q1_w = stat_record()
        self.Q1_g = stat_record()
        self.Q3_w = stat_record()
        self.Q3_g = stat_record()

    def update(self, mean_w, mean_g, max_w, max_g, range_w, range_g, Q1_w, Q1_g, Q3_w, Q3_g):
        self.mean_w.update(mean_w)
        self.mean_g.update(mean_g)
        self.max_w.update(max_w)
        self.max_g.update(max_g)
        self.range_w.update(range_w)
        self.range_g.update(range_g)
        self.Q1_w.update(Q1_w)
        self.Q1_g.update(Q1_g)
        self.Q3_w.update(Q3_w)
        self.Q3_g.update(Q3_g)
    
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, save_all=False, burn_in=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_all = save_all
        self.burn_in = burn_in

    def __call__(self, val_loss, model, path, epoch, suffix=None):
        if self.save_all:
            pn = f'checkpoint_{suffix}_ep_{epoch}.pth' if suffix is not None else f'checkpoint_ep_{epoch}.pth'
            torch.save(model.state_dict(), path  + pn)

        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, suffix)
        elif score < self.best_score + self.delta:
            if epoch > self.burn_in:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, suffix)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, suffix):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        pn = f'checkpoint_{suffix}.pth' if suffix is not None else 'checkpoint.pth'
        torch.save(model.state_dict(), path + '/' + pn)
        self.val_loss_min = val_loss
# def count_multiplications(a, b):
#     assert a.shape[1] == b.shape[1]
#     M, L = a.shape
#     N, L = b.shape
#     one1 = a != 0
#     one2 = b != 0
#     return torch.sum(torch.mul(one1.unsqueeze(1), one2)).item()

        


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def set_save_path(root, config, seed,  args):
    save_path = os.path.join(root, config)
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    writer_path = os.path.join(save_path,f'logs/seed_{seed}')

    if os.path.exists(writer_path) and os.path.isdir(writer_path):
        shutil.rmtree(writer_path)
    writer_weight = SummaryWriter(writer_path)

    save_path = os.path.join(save_path, f'runs/seed_{seed}')
    logging.info("saving to %s", save_path)
    logging.info("run arguments: %s", args)
    if os.path.exists(save_path) and os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    return writer_weight, save_path

# def get_dataset(args):
#     if args.num_clients <6:
#         if args.dataset == 'mnist':
#             test_ds = pickle.load(open('data/mnist/test_data.pk', 'rb'))
#             client_ds = pickle.load(open('data/mnist/client_data.pk', 'rb'))
#         elif args.dataset == 'thermal':
#             test_ds = pickle.load(open('data/thermal/test_data.pk', 'rb'))
#             client_ds = pickle.load(open('data/thermal/client_data.pk', 'rb'))
#             if args.training_mode == 'fp':
#                 td = []
#                 for data in test_ds:
#                     td.append((data[0] * 255/70, data[1]))
#                 test_ds = td
#                 cd = []
#                 for client in client_ds:
#                     td = []
#                     for data in client:
#                         td.append((data[0] * 255/70, data[1]))
#                     cd.append(td)
#                 client_ds = cd
#         elif args.dataset == 'climate':
#             test_ds = pickle.load(open('data/climate/test_data.pk', 'rb'))
#             client_ds = pickle.load(open('data/climate/client_data.pk', 'rb'))
#             if args.training_mode == 'fp':
#                 td = []
#                 for data in test_ds:
#                     td.append((2 * data[0] - 1, data[1]))
#                 test_ds = td
#                 cd = []
#                 for client in client_ds:
#                     td = []
#                     for data in client:
#                         td.append((2 * data[0] - 1, data[1]))
#                     cd.append(td)
#                 client_ds = cd
#         else:
#             raise ValueError(f"dataset {args.dataset} not provided")

#     else:
#         if args.dataset == 'mnist':
#             test_ds, _, client_ds, _ = prepare_mnist_data(dim=7, iid=0, 
#                                                             num_dev = args.num_clients, 
#                                                             num_ds = 300)
#         else:
#             raise ValueError(f"Only MNIST for a large number of clients")

#     return test_ds, client_ds


# def exp_setup(args, client_data):
#     if 'centralized' in args.fl:
#         args.training_mode = 'fp'
        
#     if args.dataset == 'mnist':
#         input_dim = 49
#         hidden_dim = 32
#         output_dim = 2
#         clients = [Client(input_dim, hidden_dim, output_dim, cdata, args.training_mode) for cdata in client_data]


#     elif args.dataset == 'thermal':
#         input_dim = 64
#         hidden_dim = 16
#         output_dim = 2
#         # args.local_bs = 80

#         clients = [Client(input_dim, hidden_dim, output_dim, cdata, 
#                           args.training_mode, 1, 1.5, 5.0) for cdata in client_data]
#     elif args.dataset == 'climate':
#         input_dim = 5
#         hidden_dim = 2
#         output_dim = 1
#         clients = [Client(input_dim, hidden_dim, output_dim, cdata, args.training_mode, loss=nn.MSELoss()) for cdata in client_data]

#     else:
#         raise ValueError(f"dataset {args.dataset} not provided")

#     global_model = MLP(input_dim, hidden_dim, output_dim)
#     # clients = [Client(input_dim, hidden_dim, output_dim, cdata, args.training_mode) for cdata in client_data]

#     return global_model, clients

# def find_data_iter(steps, batch_size, ds_size):
#     return int((steps * batch_size - 1) / ds_size + 1) 

# def test_inference(model, dataset, loss, no_acc = False):
#     model.eval()
#     l, correct = 0.0, 0.0

#     testloader = DataLoader(dataset, batch_size=2000, shuffle=False)
#     with torch.no_grad():
#         for i, data in enumerate(testloader):
#             x = data[0]
#             labels = data[1]
#             # Inference
#             outputs = model(x)
#             batch_loss = loss(outputs, labels)
#             l += batch_loss.item()

#             if no_acc:
#                 continue

#             # Prediction
#             _, pred_labels = torch.max(outputs, 1)
#             pred_labels = pred_labels.view(-1)
#             correct += torch.sum(torch.eq(pred_labels, labels)).item()


#     return correct/len(dataset), l



# def mnist_iid_samples(dataset, num_users, subset_size):
#     total_indices = np.arange(len(dataset))
#     np.random.shuffle(total_indices)

#     ds = []
#     sub_idx =[]
#     for i in range(num_users):
#         indices = total_indices[i*subset_size:(i+1)*subset_size]
#         sub_idx.append(indices)
#         original = [dataset[i] for i in indices]
#         ds.append([(x,label % 2, label) for (x,label) in original])
#     return ds, sub_idx


# def mnist_non_iid_samples(dataset, num_users, sub_size):
#     total_indices = np.arange(len(dataset))
#     np.random.shuffle(total_indices)

#     ds = []
#     sub_idx =[]
#     for i in range(num_users):
#         subset_label = np.random.choice(range(10), 6, replace=False)
#         subset_idx =  [i for i, (x,y) in enumerate(dataset) if y in subset_label]
#         indices = np.random.choice(subset_idx, sub_size, replace=False)
#         sub_idx.append(indices)
#         original = [dataset[i] for i in indices]
#         ds.append([(x,label % 2, label) for (x,label) in original])
#     return ds, sub_idx

    
    
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
#         return mnist_even_odd_test, mnist_even_odd_train, *mnist_iid_samples(mnist_train, num_dev, num_ds)
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
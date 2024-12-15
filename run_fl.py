#!/usr/bin/env python
from torch.utils.data import RandomSampler
from torch import nn
from data_util import *
from src.utils import *
from src.ops import *
import argparse, os, pickle, copy
from src.model import build_fp_model, build_model, nn_fp
from fl_client import *


parser = argparse.ArgumentParser()


parser.add_argument('--num_clients', type=int, default=10,
                    help="number of users: K")
parser.add_argument('--local_data', type=int, default=512,
                    help="local dataset: B")
parser.add_argument('--batch_size', type=int, default=64,
                    help="local batch size: B")
parser.add_argument('--log_interval', type=int, default=5, metavar='N',
                help='how many batches to wait before logging training status')


parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--init',
                    help='init ckpt')


parser.add_argument('--seeds', type=int, default=[0,1,2,3,4], nargs='+',
                    help='random seed (default: None)')
parser.add_argument('--asynch', action='store_true', default=False)
parser.add_argument('--niid', action='store_true', default=False)
# parser.add_argument('--save', metavar='SAVE', default='fedavg_niid/mnist',
#                     help='saved folder')
parser.add_argument('--dataset', type=str, default='femnist',
                    help='dataset dir')
parser.add_argument('--model', default=7, type=int)

parser.add_argument('--total_steps', type=int, default=80,
                    help="number of rounds of training")
parser.add_argument('--local_ep', type=int, default=12,
                    help="the number of local epochs: E")
parser.add_argument('--algorithm', choices=['FedAVG', 'FedQNN', 'FedQT', 'FedQT-BA', 'FedPAQ', 'FedPAQ-BA', 'Q-FedUpdate', 'Q-FedUpdate-BA'], default='FedQT-BA', type=str)
parser.add_argument('--qmode', default=1, type=int, help='model training: 0: NITI, 1: use int+fp calculation, 2: fp')
parser.add_argument('--quantize_comm', action='store_true', default=False)
parser.add_argument('--adaptive_bitwidth', action='store_true', default=False)
parser.add_argument('--update_mode', default=0, type=int, help='0: model update, 1: gradient update')

parser.add_argument('--initialization', default='uniform', choices=['uniform', 'normal'], type=str)
parser.add_argument('--m', default=5, type=int)

parser.add_argument('--Wbitwidth', default=4, type=int)
parser.add_argument('--Abitwidth', default=8, type=int)
parser.add_argument('--Ebitwidth', default=8, type=int)
parser.add_argument('--stochastic', action='store_true', default=True)
parser.add_argument('--use_bn', action='store_true', default=False)

parser.add_argument('--lr', type=float, default=0.05, metavar='LR')
parser.add_argument('--momentum', type=float, default=0, metavar='M')
parser.add_argument('--device', type=str, default='cuda', metavar='D',)
parser.add_argument('--num_workers', type=int, default=0, metavar='N',)

args = parser.parse_args()
logging.basicConfig(level=logging.DEBUG)
if args.device == 'cuda' and not torch.cuda.is_available():
    args.device = 'cpu'
    logging.info("Cuda is not available, using CPU")
device = torch.device(args.device)
args.device = device
b0 = 4

# args.save = f"{args.fl}_{'niid' if args.niid else 'iid'}/{args.dataset}"

def client_train(clients, global_model, num_epochs=1, batch_size=32, bitwidth_selection=None,  num_sample=None):
    updates = [] # dequantized 
    loss = []
    acc = []
    p = []
    for i, c in enumerate(clients):

        if args.qmode != 2:
            if args.Wbitwidth == 1:
                c.model_BW_hist.append(1)
            else:
                args.Wbitwidth = bitwidth_selection[i]
                c.model_BW_hist.append(bitwidth_selection[i])
        else:
            c.model_BW_hist.append(32)
        model = build_model(dataset_cfg[args.dataset]['input_channel'], 
                        dataset_cfg[args.dataset]['input_size'], 
                        dataset_cfg[args.dataset]['output_size'], 
                        args).to(args.device)
        model.load_state_dict(global_model.state_dict())
    
        if not isinstance(model, nn_fp):
            c.init_model = model.dequantize().state_dict()
        else:
            c.init_model = copy.deepcopy(model.state_dict())

        li, ai, model = c.train(model, num_epochs, batch_size, num_sample, num_workers=args.num_workers)
        p.append(len(c.train_data))


        if args.adaptive_bitwidth and bitwidth_selection is not None:
            p[-1] = p[-1] * (1 - 2.0**(1 -bitwidth_selection[i]))

        if not isinstance(model, nn_fp):
            model = model.dequantize()
        
        if args.update_mode == 0:
            updates.append(model.state_dict())
            c.train_COMM_hist.append(c.model_BW_hist[-1])
        else:
            new_state_dict = model.state_dict()
            diff = {}
            for name, param in new_state_dict.items():
                diff[name] = param - c.init_model[name]
                if 'FedPAQ' in args.algorithm:
                    scale = (diff[name].max() - diff[name].min()) / (2**bitwidth_selection[i] - 1)
                    if scale > 0:
                        Q = (diff[name] - diff[name].min()) / scale
                        floor = torch.floor(Q)
                        ceil = torch.ceil(Q)
                        Q = torch.where(torch.rand_like(Q) > (Q - floor), floor, ceil)
                        diff[name] = Q * scale + diff[name].min()
            c.train_COMM_hist.append(bitwidth_selection[i])
            updates.append(diff)
        
        if args.device == 'cuda':
            del model
            torch.cuda.empty_cache()

        loss.append(li)
        acc.append(ai)
    return updates, np.array(p)/np.sum(p), loss, acc

def average_models(models, weights, global_model_state_dict):
    state_dict = {}
    for state_dict_c, w in zip(models, weights):
        for para in state_dict_c:
            if para not in state_dict:
                state_dict[para] = state_dict_c[para]*w
            else:
                state_dict[para] += state_dict_c[para]*w
    if args.update_mode == 1:
        for para in state_dict:
            state_dict[para] += global_model_state_dict[para]
    return state_dict
    
def exp(root, config, seed):
    writer, save_path = set_save_path(root,  config, seed, args)

    set_seed(seed)

    
    train_ds_clients, test_ds_clients, test_ds  = get_fl_dataset(args, args.local_data, args.num_clients)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=args.num_workers)
    
    
    global_model = build_fp_model(dataset_cfg[args.dataset]['input_channel'], 
                        dataset_cfg[args.dataset]['input_size'], 
                        dataset_cfg[args.dataset]['output_size'], args.model, args.lr, args.device)
    args.num_clients = len(train_ds_clients)
    if args.init:
        logging.info("Init weights from: %s", args.init)
        global_model.load_state_dict(torch.load(args.init))
    
    criterion = nn.CrossEntropyLoss()
    
    server = Server(global_model)
    bitwidth_limits = np.random.randint(b0, 17, args.num_clients)
    with open(save_path+'/bitwidth_limit.txt', 'w') as f:
        f.write(str(bitwidth_limits) + '\n')
    clients = [Client(i, train_data, eval_data, criterion, save_path, bitwidth_limits[i]) 
               for i, train_data, eval_data in zip(list(range(args.num_clients)), train_ds_clients, test_ds_clients)]
    
    print(server.get_clients_info(clients)[2])
    tloss, tacc, vloss, vacc, comm = [], [], [], [], []
    cum_comm = 0
    best_acc = 0

    if args.dataset == 'mnist':
        threshold_acc = 95
    elif args.dataset == 'cifar10':
        threshold_acc = 60
    else: # FEMNIST
        threshold_acc = 82
    average_epoch = None
    average_model_size = None
    average_comm_size = None
    val_loss, _ = global_model.epoch(test_loader, 0, args.log_interval, criterion, train=False)
    f0 = val_loss
    C = 16/(np.log2(f0+1))
    for steps in range(0, args.total_steps):
        print(f" ----------- Step {steps} -----------------")

        number_of_clients = int(np.floor(args.num_clients * 0.2))

        server.select_clients_random(steps, clients, number_of_clients)

        if not args.quantize_comm and args.qmode != 2:
            bitwidth_selecton = None

        elif args.adaptive_bitwidth:
            bm = b0 + int(np.ceil(C * np.log2(max(1.0, (f0+1)/(val_loss + 1)))))
            print(f"bm: {bm}")
            bitwidth_selecton = [min(bm, c.bitwidth_limit) for c in clients]
            # p = [(1 - 2.0**(2-bn)) for bn in bitwidth_selecton]
            # server.select_clients_random(steps, clients, number_of_clients, prob=np.array(p)/np.sum(p))
        else:
            bitwidth_selecton = [c.bitwidth_limit for c in server.selected_clients]
        # selected_clients = server.update_client_model(server.selected_clients)
        updates, weights, loss, acc = client_train(server.selected_clients, global_model, args.local_ep, args.batch_size, 
                                          bitwidth_selection=bitwidth_selecton)

        averged_update = average_models(updates, weights, global_model.state_dict())
        global_model.load_state_dict(averged_update)
        cum_comm += sum([c.train_COMM_hist[-1] for c in server.selected_clients])
        val_loss, val_prec1= global_model.epoch(test_loader, steps, args.log_interval, criterion, train=False)

        writer.add_scalar('Accuracy/train', np.average(acc), steps)
        writer.add_scalar('Loss/train', np.average(loss), steps)

        writer.add_scalar("Global/Loss/test", val_loss, steps)
        writer.add_scalar("Global/Acc/test", val_prec1, steps)
        
        best_acc = max(val_prec1, best_acc)
        writer.add_scalar('Accuracy/best', best_acc, steps)
        if best_acc > threshold_acc and average_epoch is None:
            average_epoch = [c.train_epoch for c in clients]
            average_model_size = [bs  for c in clients for bs in c.model_BW_hist]
            average_comm_size = [bs for c in clients for bs in c.train_COMM_hist]
            logging.info("# of Epochs: %s", average_epoch)
            logging.info("# of Bitwidth: %s", average_model_size)

        logging.info("best_acc: %f %s", best_acc ,save_path)
        logging.info('Epoch: {0} '
                     'Train Acc {train_prec1:.3f} '
                     'Train Loss {train_loss:.3f} '
                     'Valid Acc {val_prec1:.3f} '
                     'Valid Loss {val_loss:.3f} \n'
                     .format(steps,
                             train_prec1=np.average(acc), val_prec1=val_prec1,
                             train_loss=np.average(loss), val_loss=val_loss))
        tloss.append(loss)
        tacc.append(acc)
        vloss.append(val_loss)
        vacc.append(val_prec1)
        comm.append(cum_comm)
        # output = root + '/' + config + f'/temp/temp_{seed}_{steps}.pk'
        # pickle.dump([tloss, tacc, vloss, vacc, comm,
        #              [c.train_epoch for c in clients], 
        #              [bs  for c in clients for bs in c.model_BW_hist],
        #              [bs for c in clients for bs in c.train_COMM_hist]
        #              ], open(output, 'wb'))
        # torch.save(global_model.state_dict(), root + '/' + config + f'/temp/model_{seed}_{steps}.pk')
        
    
    if average_epoch is None:
        average_epoch = [c.train_epoch for c in clients]
        average_model_size = [bs  for c in clients for bs in c.model_BW_hist]
        average_comm_size = [bs for c in clients for bs in c.train_COMM_hist]

    return tloss, tacc, vloss, vacc, comm, average_epoch, average_model_size, average_comm_size



if __name__ == '__main__':
    config = args.algorithm

    root = './results/' 
    if not os.path.exists(root):
        os.makedirs(root)
    
    root += 'async_' if args.asynch else 'sync_'
    root += f'niid_{args.local_ep}' if args.niid else f'iid_{args.local_ep}'
    root += f'_{args.dataset}'
    if not os.path.exists(root):
        os.makedirs(root)

    path = root + '/' + config
    if not os.path.exists(path):
        os.makedirs(path)
    
    temp = path + '/temp'
    if not os.path.exists(temp):
        os.makedirs(temp)

    if args.algorithm == 'FedAVG':
        args.qmode = 2
    elif args.algorithm == 'FedPAQ':
        args.qmode = 2
        args.update_mode = 1
        args.quantize_comm = True
    elif args.algorithm == 'FedPAQ-BA':
        args.qmode = 2
        args.update_mode = 1
        args.quantize_comm = True
        args.adaptive_bitwidth = True
    elif args.algorithm == 'FedQNN':
        args.qmode = 0
        args.update_mode = 1
        args.Wbitwidth = 1
    elif args.algorithm == 'Q-FedUpdate':
        args.qmode = 0
        args.update_mode = 1
    elif args.algorithm == 'Q-FedUpdate-BA':
        args.qmode = 0
        args.update_mode = 1
        args.adaptive_bitwidth = True
    elif args.algorithm == 'FedQT':
        args.qmode = 1
        args.update_mode = 0
    elif args.algorithm == 'FedQT-BA':
        args.qmode = 1
        args.update_mode = 0
        args.adaptive_bitwidth = True
    else:
        raise ValueError("Algorithm not supported")
    
    if args.qmode != 2:
        args.quantize_comm = True

    with open(path+'/args.txt', 'w') as f:
        f.write(str(args) + '\n')
    
    for seed in args.seeds:
        output = path + f'/results_seed_{seed}.pk'
        tloss, tacc, vloss, vacc, comm, comp, mem, comm_each = exp(root, config, seed)
        pickle.dump([tloss, tacc, vloss, vacc, comm, comp, mem, comm_each], open(output, 'wb'))
        file_path = path + '/runs'
        try:
            shutil.rmtree(file_path)
            print(f"File '{file_path}' deleted successfully.")
        except FileNotFoundError:
            print(f"File '{file_path}' not found.")
        except PermissionError:
            print(f"You don't have permission to delete '{file_path}'.")
        except Exception as e:
            print(f"An error occurred: {e}")

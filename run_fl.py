#!/usr/bin/env python
from torch.utils.data import RandomSampler
from torch import nn
from data_util import *
from src.utils import *
import argparse, os, pickle
from src.model import build_fp_model, build_model, nn_fp
from fl_client import *

parser = argparse.ArgumentParser()

parser.add_argument('--total_steps', type=int, default=70,
                    help="number of rounds of training")
parser.add_argument('--num_clients', type=int, default=10,
                    help="number of users: K")
parser.add_argument('--local_ep', type=int, default=6,
                    help="the number of local epochs: E")
parser.add_argument('--local_data', type=int, default=512,
                    help="local dataset: B")
parser.add_argument('--batch_size', type=int, default=64,
                    help="local batch size: B")
parser.add_argument('--log_interval', type=int, default=5, metavar='N',
                help='how many batches to wait before logging training status')

parser.add_argument('--fl', default='fedavg', type=str)
parser.add_argument('--niid', action='store_true', default=False)
parser.add_argument('--adaptive_bitwidth', action='store_true', default=False)

parser.add_argument('--save', metavar='SAVE', default='fedavg_niid/mnist',
                    help='saved folder')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--init',
                    help='init ckpt')
parser.add_argument('--seeds', type=int, default=[2], nargs='+',
                    help='random seed (default: None)')

parser.add_argument('--model', default=4, type=int)
parser.add_argument('--qmode', default=1, type=int, help='0: NITI, 1: use int+fp calculation, 2: fp')
parser.add_argument('--dataset', type=str, default='mnist',
                    help='dataset dir')
parser.add_argument('--initialization', default='uniform', choices=['uniform', 'normal'], type=str)
parser.add_argument('--m', default=5, type=int)

parser.add_argument('--Wbitwidth', default=4, type=int)
parser.add_argument('--Abitwidth', default=8, type=int)
parser.add_argument('--Ebitwidth', default=8, type=int)
parser.add_argument('--stochastic', action='store_true', default=True)
parser.add_argument('--use_bn', action='store_true', default=False)

parser.add_argument('--lr', type=float, default=0.05, metavar='LR')
parser.add_argument('--momentum', type=float, default=0, metavar='M')
parser.add_argument('--device', type=str, default='cpu', metavar='D',)
parser.add_argument('--num_workers', type=int, default=0, metavar='N',)

args = parser.parse_args()
logging.basicConfig(level=logging.DEBUG)
if args.device == 'cuda' and not torch.cuda.is_available():
    args.device = 'cpu'
    logging.info("Cuda is not available, using CPU")
device = torch.device(args.device)
args.device = device
b0 = 4


args.save = f"{args.fl}_{'niid' if args.niid else 'iid'}/{args.dataset}"

def client_train(clients, num_epochs=1, batch_size=32, adaptive=False, bitwidth_selection=None,  num_sample=None):
    updates = [] # dequantized 
    loss = []
    acc = []
    p = []
    for i, c in enumerate(clients):
        if args.qmode != 2  :
            args.Wbitwidth = bitwidth_selection[i]
            c.train_bitwidth_hist.append(bitwidth_selection[i])
        else:
            c.train_bitwidth_hist.append(32)
        model = build_model(dataset_cfg[args.dataset]['input_channel'], 
                        dataset_cfg[args.dataset]['input_size'], 
                        dataset_cfg[args.dataset]['output_size'], 
                        args)
        model.load_state_dict(torch.load(c.model_path))

        
        li, ai, model = c.train(model, num_epochs, batch_size, num_sample, num_workers=args.num_workers)
        p.append(len(c.train_data))
        if args.adaptive_bitwidth and bitwidth_selection is not None:
            p[-1] = p[-1] * (1 - 2.0**(1 -bitwidth_selection[i]))

        if not isinstance(model, nn_fp):
            model = model.dequantize()
        updates.append(model.state_dict())
        loss.append(li)
        acc.append(ai)
    return updates, np.array(p)/np.sum(p), loss, acc

def average_models(models, weights):
    state_dict = {}
    for state_dict_c, w in zip(models, weights):
        for para in state_dict_c:
            if para not in state_dict:
                state_dict[para] = state_dict_c[para]*w
            else:
                state_dict[para] += state_dict_c[para]*w
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
    tloss, tacc, vloss, vacc = [], [], [], []
    best_acc = 0
    threshold_acc = 95
    average_epoch = None
    average_model_size = None
    val_loss, _ = global_model.epoch(test_loader, 0, args.log_interval, criterion, train=False)
    f0 = val_loss
    C = 16/(np.log2(f0+1))
    for steps in range(0, args.total_steps):
        print(f"Step {steps}")

        number_of_clients = int(np.floor(args.num_clients * 0.2))

        server.select_clients_random(steps, clients, number_of_clients)

        if args.qmode == 2:
            bitwidth_selecton = None

        elif args.adaptive_bitwidth:
            bm = b0 + int(np.floor(C * np.log2(max(1.0, (f0+1)/(val_loss + 1)))))
            print(f"bm: {bm}")
            bitwidth_selecton = [min(bm, c.bitwidth_limit) for c in clients]
            # p = [(1 - 2.0**(2-bn)) for bn in bitwidth_selecton]
            # server.select_clients_random(steps, clients, number_of_clients, prob=np.array(p)/np.sum(p))
        else:
            bitwidth_selecton = [c.bitwidth_limit for c in server.selected_clients]
        selected_clients = server.update_client_model(server.selected_clients)
        updates, weights, loss, acc = client_train(server.selected_clients, args.local_ep, args.batch_size, adaptive=args.adaptive_bitwidth,
                                          bitwidth_selection=bitwidth_selecton)

        averged_update = average_models(updates, weights)
        global_model.load_state_dict(averged_update)
        val_loss, val_prec1= global_model.epoch(test_loader, steps, args.log_interval, criterion, train=False)

        writer.add_scalar('Accuracy/train', np.average(acc), steps)
        writer.add_scalar('Loss/train', np.average(loss), steps)

        writer.add_scalar("Global/Loss/test", val_loss, steps)
        writer.add_scalar("Global/Acc/test", val_prec1, steps)
        
        best_acc = max(val_prec1, best_acc)
        writer.add_scalar('Accuracy/best', best_acc, steps)
        if best_acc > threshold_acc and average_epoch is None:
            average_epoch = [c.train_epoch for c in clients]
            average_model_size = [np.mean(c.train_bitwidth_hist) for c in clients]
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
        output = root + '/' + config + f'/temp_{seed}_{steps}.pk'
        pickle.dump([tloss, tacc, vloss, vacc, 
                     [c.train_epoch for c in clients], [c.train_bitwidth_hist for c in clients]], open(output, 'wb'))
        torch.save(global_model.model.state_dict(), root + '/' + config + f'/temp_model_{seed}_{steps}.pk')
        
    
    if average_epoch is None:
        average_epoch = [c.train_epoch for c in clients]
        average_model_size = [np.mean(c.train_bitwidth_hist) for c in clients]

    return tloss, tacc, vloss, vacc, average_epoch, average_model_size



if __name__ == '__main__':
    config = f'{args.dataset}_model_{args.model}'
    if args.qmode == 2:
        config += f'_fp_lr_{args.lr}'
    elif args.qmode == 0:
        config += f'_NITI_m_{args.m}_adaptive_{args.adaptive_bitwidth}'
    else:
        config += f'_quant_lr_{args.lr}_adaptive_{args.adaptive_bitwidth}'


    root = './fl_results/'+args.save 
    if not os.path.exists(root):
        os.makedirs(root)

    path = root + '/' + config 
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+'/args.txt', 'w') as f:
        f.write(str(args) + '\n')
    
    for seed in args.seeds:
        output = path + f'/results_seed_{seed}.pk'
        tloss, tacc, vloss, vacc, comp, mem = exp(root, config, seed)
        pickle.dump([tloss, tacc, vloss, vacc, comp, mem], open(output, 'wb'))

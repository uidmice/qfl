from torch.utils.data import RandomSampler
from torch import nn
from data_util import *
from src.utils import *
from src.ops import *
import argparse, os, pickle, copy
from src.model import build_fp_model, build_model, nn_fp
from fl_client import *
from torch.utils.data import ConcatDataset


parser = argparse.ArgumentParser()


parser.add_argument('--num_clients', type=int, default=10,
                    help="number of users: K")
parser.add_argument('--local_data', type=int, default=512,
                    help="local dataset: B")
parser.add_argument('--batch_size', type=int, default=64,
                    help="local batch size: B")
parser.add_argument('--num_batch', type=int, default=80,
                    help="local batch size: B")
parser.add_argument('--log_interval', type=int, default=5, metavar='N',
                help='how many batches to wait before logging training status')


parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--init',
                    help='init ckpt')


parser.add_argument('--seeds', type=int, default=[0], nargs='+',
                    help='random seed (default: None)')
parser.add_argument('--asynch', action='store_true', default=False)
parser.add_argument('--niid', action='store_true', default=False)
# parser.add_argument('--save', metavar='SAVE', default='fedavg_niid/mnist',
#                     help='saved folder')
parser.add_argument('--dataset', type=str, default='mnist',
                    help='dataset dir')
parser.add_argument('--model', default=4, type=int)

parser.add_argument('--total_steps', type=int, default=200,
                    help="number of rounds of training")
parser.add_argument('--local_ep', type=int, default=1,
                    help="the number of local epochs: E")
parser.add_argument('--algorithm', choices=['FedAVG', 'FedQNN', 'FedQT', 'FedQT-BA', 'FedPAQ', 'FedPAQ-BA', 'Q-FedUpdate', 'Q-FedUpdate-BA'], default='FedAVG', type=str)
parser.add_argument('--qmode', default=0, type=int, help='model training: 0: NITI, 1: use int+fp calculation, 2: fp')
parser.add_argument('--quantize_comm', action='store_true', default=False)
parser.add_argument('--adaptive_bitwidth', action='store_true', default=False)
parser.add_argument('--update_mode', default=0, type=int, help='0: model update, 1: gradient update')

parser.add_argument('--initialization', default='uniform', choices=['uniform', 'normal'], type=str)
parser.add_argument('--m', default=5, type=int)

parser.add_argument('--Wbitwidth', default=6, type=int)
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


def exp(root, seed):
    
    set_seed(seed)

    fp_model = build_fp_model(dataset_cfg[args.dataset]['input_channel'], 
                        dataset_cfg[args.dataset]['input_size'], 
                        dataset_cfg[args.dataset]['output_size'], args.model, args.lr, args.device)

    model = build_model(dataset_cfg[args.dataset]['input_channel'], 
                        dataset_cfg[args.dataset]['input_size'], 
                        dataset_cfg[args.dataset]['output_size'], 
                        args).to(args.device)
    
    model.load_state_dict(fp_model.state_dict())

    best_prec1 = 0

    criterion = nn.CrossEntropyLoss()

    train_ds_clients, test_ds_clients, test_ds  = get_fl_dataset(args, 5000, 1)
    train_ds = ConcatDataset(train_ds_clients)
    random_sample = RandomSampler(train_ds, num_samples=args.num_batch * args.batch_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=random_sample)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    


    tloss, tacc, vloss, vacc = [], [], [], []

    for epoch in range(0, args.total_steps):
        train_loss, train_prec1= model.epoch(train_loader, epoch, args.log_interval, criterion, train=True)

        val_loss, val_prec1= model.epoch(test_loader, epoch, args.log_interval, criterion, train=False)

        tloss.append(train_loss)
        tacc.append(train_prec1)
        vloss.append(val_loss)
        vacc.append(val_prec1)

        best_prec1 = max(val_prec1, best_prec1)


        logging.info("best_acc: %f", best_prec1)
        logging.info('Epoch: {0} '
                     'Train Acc {train_prec1:.3f} '
                     'Train Loss {train_loss:.3f} '
                     'Valid Acc {val_prec1:.3f} '
                     'Valid Loss {val_loss:.3f} \n'
                     .format(epoch,
                             train_prec1=train_prec1, val_prec1=val_prec1,
                             train_loss=train_loss, val_loss=val_loss))
    return tloss, tacc, vloss, vacc


if __name__ == '__main__':



    root = './results_qt'
    if not os.path.exists(root):
        os.makedirs(root)
    
    
    root = root + '/' + f'model_{args.model}/'
    if not os.path.exists(root):
        os.makedirs(root)

    if args.qmode == 2:
        root += f'fp'
    elif args.qmode == 0:
        root += f'NITI_bitwidth_{args.Wbitwidth}'
    else:
        root += f'quant_bitwidth_{args.Wbitwidth}'
    
    if not os.path.exists(root):
        os.makedirs(root)

    with open(root+'/args.txt', 'w') as f:
        f.write(str(args) + '\n')
    
    for seed in args.seeds:
        output = root + f'/results_seed_{seed}.pk'
        tloss, tacc, vloss, vacc = exp(root,  seed)
        pickle.dump([tloss, tacc, vloss, vacc], open(output, 'wb'))

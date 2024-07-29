from torch.utils.data import RandomSampler
from torch import nn
from data_util import *
from src.utils import *
import argparse, os, pickle
from src.model import build_model

parser = argparse.ArgumentParser()
parser.add_argument('--save', metavar='SAVE', default='test',
                    help='saved folder')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--init',
                    help='init ckpt')
parser.add_argument('--seeds', type=int, default=[0,1,2,3,4,5], nargs='+',
                    help='random seed (default: None)')
parser.add_argument('--max_epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--burn_in', default=100, type=int, metavar='N',
                    help='number of min epochs to run')
parser.add_argument('--batch_number', type=int, default=20,
                    help="local batch number: B")
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--model', default=4, type=int)
parser.add_argument('--qmode', default=0, type=int, help='0: NITI, 1: use int+fp calculation, 2: fp')
parser.add_argument('--dataset', type=str, default='mnist',
                    help='dataset dir')
parser.add_argument('--initialization', default='uniform', choices=['uniform', 'normal'], type=str)
parser.add_argument('--m', default=3, type=int)

parser.add_argument('--Wbitwidth', default=15, type=int)
parser.add_argument('--Abitwidth', default=8, type=int)
parser.add_argument('--Ebitwidth', default=8, type=int)
parser.add_argument('--stochastic', action='store_true', default=False)
parser.add_argument('--use_bn', action='store_true', default=False)

parser.add_argument('--lr', type=float, default=0.02, metavar='LR')
parser.add_argument('--momentum', type=float, default=0, metavar='M',
                    help='SGD momentum (default: 0.8)')


args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG)


def exp(root, config, seed):
    writer_weight, save_path = set_save_path(root,  config, seed, args)
    
    set_seed(seed)

    model = build_model(dataset_cfg[args.dataset]['input_channel'], 
                        dataset_cfg[args.dataset]['input_size'], 
                        dataset_cfg[args.dataset]['output_size'], 
                        args)
    if args.init:
        logging.info("Init weights from: %s", args.init)
        model.load_state_dict(torch.load(args.init))

    early_stopping = EarlyStopping(patience=7, verbose=True, burn_in=args.max_epochs)
    best_prec1 = 0

    if args.dataset == 'spectrum' or args.dataset == 'CWRU':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    train_ds, test_ds = get_dataset(args)
    random_sample = RandomSampler(train_ds, num_samples=args.batch_number * args.batch_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=random_sample)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    tloss, tacc, vloss, vacc = [], [], [], []

    for epoch in range(0, args.max_epochs):
        train_loss, train_prec1= model.epoch(train_loader, epoch, args.log_interval, criterion, train=True)
        writer_weight.add_scalar('Accuracy/train', train_prec1, epoch)
        writer_weight.add_scalar('Loss/train', train_loss, epoch)

        val_loss, val_prec1= model.epoch(test_loader, epoch, args.log_interval, criterion, train=False)
        writer_weight.add_scalar('Accuracy/test', val_prec1, epoch)
        writer_weight.add_scalar('Loss/test', val_loss, epoch)

        tloss.append(train_loss)
        tacc.append(train_prec1)
        vloss.append(val_loss)
        vacc.append(val_prec1)

        best_prec1 = max(val_prec1, best_prec1)

        writer_weight.add_scalar('Accuracy/best', best_prec1, epoch)

        logging.info("best_acc: %f %s", best_prec1,save_path)
        early_stopping(val_loss, model, save_path, epoch)
        logging.info('Epoch: {0} '
                     'Train Acc {train_prec1:.3f} '
                     'Train Loss {train_loss:.3f} '
                     'Valid Acc {val_prec1:.3f} '
                     'Valid Loss {val_loss:.3f} \n'
                     .format(epoch,
                             train_prec1=train_prec1, val_prec1=val_prec1,
                             train_loss=train_loss, val_loss=val_loss))
        if epoch % 20 == 0:
            pn = f'checkpoint_{epoch}.pth'
            torch.save(model.state_dict(), save_path + '/' + pn)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    return tloss, tacc, vloss, vacc


if __name__ == '__main__':

    config = f'{args.dataset}_model_{args.model}'
    if args.qmode == 2:
        config += f'_fp_lr_{args.lr}'
    elif args.qmode == 0:
        config += f'_NITI_bitwidth_{args.Wbitwidth}_{args.Abitwidth}_{args.Ebitwidth}_{args.m}'
    else:
        config += f'_quant_bitwidth_{args.Wbitwidth}_{args.Abitwidth}_{args.Ebitwidth}_lr_{args.lr}'
        if args.stochastic:
            config += f'_sr_{args.stochastic}'

    root = './pre_results/'+args.save 
    if not os.path.exists(root):
        os.makedirs(root)

    path = root + '/' + config 
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+'/args.txt', 'w') as f:
        f.write(str(args) + '\n')
    
    for seed in args.seeds:
        output = path + f'/results_seed_{seed}.pk'
        tloss, tacc, vloss, vacc = exp(root, config, seed)
        pickle.dump([tloss, tacc, vloss, vacc], open(output, 'wb'))


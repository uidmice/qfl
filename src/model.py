import torch
import torch.nn as nn
import torch.optim as optim
from  src.qm import  *
import collections, time
from src.ops import *
from src.meters import accuracy, AverageMeter

model_dict = {
    0: [[32, 3, 1], [64, 3, 1], "M",'F','D', 128],
    8: [[64,3, 1], [64, 3, 1], 'M', [128,3,1], [128,3,1], [128,3,1], 'M', 
        'F', 'D', 128],
    1: [[128,3, 1], [128, 3, 1], 'M', [256,3,1], [256,3,1], 'M', [512,3,1], [512, 3,1], 'M', 
        'F', 'D', 128],
    2: [[128,3, 1], [128,3, 1], [128,3, 1], 'M', [256,3,1], [256,3,1], 'M', [512,3,1], [512, 3,1], 'M', 
        'F', 'D', 128],
    3: [128, 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 'M', 128],
    4: [[6, 5, 2], "M", [16, 5, 0], "M", 'F', 120, 84 ], # LeNet 5, mnist
    5: [24,24],
    6: [32, 16, 32],
    7: [[32, 5, 2],"M", [64, 5, 2], "M", 'F', 2048], # leaf femnist cnn
}


class Qnet(nn.Module):

    def __init__(self, loss, quantizer):
        super(Qnet, self).__init__()
        self.loss = loss
        self.quantizer = quantizer

    def forward(self, x):
        x_data, scale = self.quantizer(x)
        x = x_data, scale
        print(self.forward_layers.device)
        print(x_data.device)
        self.out, self.out_s = self.forward_layers(x)
        return self.out, self.out_s

    def backward(self, target):
        # x = self.loss(self.out, self.out_s, target)
        x, s = self.loss(self.out, self.out_s, target)
        x = x, s
        for layer in reversed(self.forward_layers):
            if isinstance(layer,torch.nn.Sequential):
                for block in reversed(layer):
                    x=block.backward(x)
            else:
                x=layer.backward(x)

    def state_dict(self):
        state_dict=collections.OrderedDict()
        for idx,l in enumerate(self.forward_layers):
            if hasattr(l,'weight'):
                layer_prefix = 'layers.'+str(idx)+'.'
                state_dict[layer_prefix+'weight']=l.weight
                state_dict[layer_prefix+'weight_scale']=l.weight_scale[0]
        return state_dict

    def load_state_dict(self, state_dict): #load fp state_dict
        for idx,l in enumerate(self.forward_layers):
            if hasattr(l,'weight'):
                layer_prefix = 'layers.'+str(idx)+'.'
                l.weight = state_dict[layer_prefix+'weight']
                l.weight, l.weight_scale=l.quantizer(l.weight)

class nn_q(Qnet):
    def __init__(self, channel, img_size, out_dim, cfg, loss, 
                 weight_update, forward_shift, backward_shift, input_quantizer, quantizer, initialize, device, use_bias=False):
        super(nn_q, self).__init__(loss, input_quantizer)
        self.channel = channel
        self.img_size = img_size
        self.out_dim = out_dim
        self.cfg = cfg
        self.device = device

        layers = []
        ldim = 0
        for x in cfg:
            if x == 'M':
                layers += [QMaxpool2d(kernel_size=2, stride=2)]
                img_size = img_size // 2
            elif isinstance(x, list):
                layers += [QConv2d(channel, x[0], kernel_size=x[1], stride=1, padding=x[2], quantizer=quantizer, 
                                   weight_update=weight_update, initialize=initialize),
                           QReLU(forward_shift, backward_shift)]
                channel = x[0]
                img_size = (img_size + 2*x[2] - x[1]) + 1
            elif x == 'F':
                layers += [QFlat()]
            elif x == 'D':
                layers += [QDropout(0.1)]
            else:
                if ldim == 0:
                    if img_size > 0:
                        layers += [QLinear(channel * img_size * img_size, x, quantizer, weight_update, initialize, bias=use_bias),
                            QReLU(forward_shift, backward_shift)]
                    else:
                        layers += [QLinear(channel, x, quantizer, weight_update, initialize,bias=use_bias),
                            QReLU(forward_shift, backward_shift)]
                    ldim = x
                else:
                    layers += [QLinear(ldim, x, quantizer, weight_update, initialize,bias=use_bias),
                           QReLU(forward_shift, backward_shift)]
                    ldim = x

        layers += [
            QLinear(ldim, out_dim, quantizer, weight_update, initialize,bias=use_bias)
        ]
        self.use_bias = use_bias
        self.forward_layers = nn.Sequential(*layers).to(device)

    def dequantize(self):
        fp_model = nn_fp(self.channel, self.img_size, self.out_dim, self.cfg, self.device, bias=self.use_bias)
        state_dict = self.state_dict()
        new_dict = {}
        for idx,l in enumerate(fp_model.layers):
            if hasattr(l,'weight'):
                layer_prefix = 'layers.'+str(idx)+'.'
                data = state_dict[layer_prefix+'weight'] * state_dict[layer_prefix+'weight_scale']
                if l.bias:
                    new_dict[layer_prefix+'weight']=data[:,:-1]
                    new_dict[layer_prefix+'bias']=data[:,-1]
                else:
                    new_dict[layer_prefix+'weight']=data
        fp_model.load_state_dict(new_dict)
        return fp_model
    

    def epoch(self, data_loader, epoch, log_interval, criterion, train=True):
        if train:
            self.train()
        else:
            self.eval()
        loss_meter, acc_meter, time_meter = AverageMeter(), AverageMeter(), AverageMeter()
        start_time = time.time()
        for batch_idx, (inputs, target) in enumerate(data_loader):
            output, output_s = self.forward(inputs.to(self.device))
            output_s = output_s[0].cpu()
            loss = criterion(output.float().cpu()*output_s, target)
            loss_meter.update(float(loss), inputs.size(0))
            if  isinstance(criterion, nn.CrossEntropyLoss):
                acc = accuracy(output.float().cpu()*output_s, target)
                acc_meter.update(float(acc), inputs.size(0))

            time_meter.update(time.time() - start_time)
            start_time = time.time()

            if train:
                self.backward(target.to(self.device))
                if log_interval > 0 and batch_idx % log_interval == 0:
                    print('[{0}][{1:>3}/{2}] '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'loss {loss.val:.3f} ({loss.avg:.3f}) '
                        'acc {top1.val:.3f} ({top1.avg:.3f}) '
                        .format(
                            epoch, batch_idx, len(data_loader),
                            batch_time=time_meter,
                            loss=loss_meter,
                            top1=acc_meter))
        return loss_meter.avg, acc_meter.avg


class nn_fp(nn.Module):
    def __init__(self, channel, img_size, out_dim, cfg, device, lr=0.01, momentum=0.9, bias=False):
        super(nn_fp, self).__init__()
        self.channel = channel
        self.img_size = img_size
        self.out_dim = out_dim
        self.cfg = cfg
        layers = []
        ldim = 0
        self.device = device
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                img_size = img_size // 2
            elif isinstance(x, list):
                layers += [nn.Conv2d(channel, x[0], kernel_size=x[1], padding=x[2], bias=False),
                           nn.ReLU()]
                channel = x[0]
                img_size = (img_size + 2*x[2] - x[1]) + 1
            elif x == 'F':
                layers += [nn.Flatten()]
            elif x == 'D':
                layers += [nn.Dropout(0.5)]
            else:
                if ldim == 0:
                    if img_size > 0:
                        layers += [nn.Linear(channel * img_size * img_size, x, bias = bias),
                           nn.ReLU()]
                    else:
                        layers += [nn.Linear(channel, x, bias = bias),
                           nn.ReLU()]
                    ldim = x
                else:
                    layers += [nn.Linear(ldim, x, bias = bias),
                           nn.ReLU()]
                    ldim = x

        layers += [
            nn.Linear(ldim, out_dim,bias=False)
        ]
        self.layers = nn.Sequential(*layers).to(device)
        self.setup_optimizer(lr, momentum)

    def setup_optimizer(self, lr, momentum):
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def forward(self, x):
        return self.layers(x)
    
    def epoch(self, data_loader, epoch, log_interval, criterion, train=True):
        if train:
            self.train()
        else:
            self.eval()

        loss_meter, acc_meter, time_meter = AverageMeter(), AverageMeter(), AverageMeter()
        start_time = time.time()
        for batch_idx, (inputs, target) in enumerate(data_loader):
            self.optimizer.zero_grad()
            output = self.forward(inputs.to(self.device)).cpu()
            loss = criterion(output, target)
            loss_meter.update(float(loss.item()), inputs.size(0))
            if  isinstance(criterion, nn.CrossEntropyLoss):
                acc = accuracy(output.cpu(), target)
                acc_meter.update(float(acc), inputs.size(0))
            time_meter.update(time.time() - start_time)
            start_time = time.time()

            if train:
                loss.backward()
                self.optimizer.step()
                if log_interval>0 and batch_idx % log_interval == 0:
                    print('[{0}][{1:>3}/{2}] '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'loss {loss.val:.3f} ({loss.avg:.3f}) '
                        'acc {top1.val:.3f} ({top1.avg:.3f}) '
                        .format(
                            epoch, batch_idx, len(data_loader),
                            batch_time=time_meter,
                            loss=loss_meter,
                            top1=acc_meter))
        return loss_meter.avg, acc_meter.avg
    



def NITI_weight_update(w, ws, g, gs, m, range):
    int32_bitwidth = get_bitwidth(g)
    if int32_bitwidth == 0:
        return int8_clip(w,range)
    shift = int32_bitwidth - m
    if shift > 1:
        g = StoShift(g, shift.int().item())
    w = w - g
    return int8_clip(w, range)
 
    
def fp_weight_update(w, ws, g, gs, bitwidth, bs, lr):
    wn = w * ws[0] - g * gs[0] * lr/bs
    wt, scale = fp_quant_stochastic(wn, bitwidth)
    ws[0] = scale[0]
    return int8_clip(wt, 2**bitwidth - 1)

def build_fp_model(in_channel, img_dim, out_dim, model_id, lr, device, momentum=0, use_bn=False):
    cfg = model_dict[model_id]
    model = model = nn_fp(in_channel, img_dim, out_dim, cfg, device, lr, momentum, use_bn)
    return model

def build_q_model(in_channel, img_dim, out_dim, model_id, Wb, batch_size, lr, device, Ab=8, Eb=8, stochastic=True,loss='CE'):
    cfg = model_dict[model_id]
    w_quant = lambda x: fp_quant(x, Wb - 1)
    if stochastic:
        e_quant = lambda x: fp_quant_stochastic(x, Eb - 1)
        a_quant = lambda x: fp_quant_stochastic(x, Ab - 1)
    else:
        e_quant = lambda x: fp_quant(x, Eb - 1)
        a_quant = lambda x: fp_quant(x, Ab - 1)
    
    if loss == 'CE':
        loss = QCELoss(e_quant)
    else:
        loss = QMSELoss(e_quant)
        
    weight_update = lambda a, b, c, d: fp_weight_update(a,b,c,d, Wb - 1, batch_size, lr)
    a_rescale = lambda a, s: a_quant(a*s[0])
    e_rescale = lambda a, s: e_quant(a*s[0])
    model = nn_q(in_channel, img_dim, out_dim, cfg, loss, weight_update, a_rescale, e_rescale, 
                a_quant, w_quant, 'uniform', device, use_bias=False)
    return model
    

def build_NITI_model(in_channel, img_dim, out_dim, model_id, Wb, device, Ab=8, Eb=8, m=5, loss='CE'):
    cfg = model_dict[model_id]
    if Wb == 1:
        w_quant = lambda x: torch.where(x > 0, torch.tensor(1), torch.tensor(-1)), [1.0]
        weight_update = lambda a,b,c,d: NITI_weight_update(a,b,c,d, 7, 2**7-1)

    else:
        w_quant = lambda x: fp_quant(x, Wb - 1)
        weight_update = lambda a,b,c,d: NITI_weight_update(a,b,c,d, Wb - m, 2**(Wb-1)-1)

    a_quant = lambda x: fp_quant(x, Ab - 1)
    e_quant = lambda x: fp_quant(x, Eb - 1)
    if loss == 'CE':
        loss = QCELoss(e_quant)
    else:
        loss = QMSELoss(e_quant)

    a_shit = lambda a, s: shift(a,s, Ab - 1)
    e_shift = lambda a, s : shift(a, s, Eb - 1)
    model = nn_q(
        in_channel, img_dim, out_dim, cfg, loss, weight_update, a_shit, e_shift, 
                a_quant, w_quant, 'uniform',  device, use_bias=False)
    return model

def build_model(in_channel, img_dim, out_dim, args):
    if args.qmode == 2:
        model = build_fp_model(in_channel, img_dim, out_dim, args.model, args.lr, args.device, momentum=args.momentum, use_bn=False)
    else:
        loss = 'CE'
        if args.dataset == 'spectrum' or args.dataset == 'CWRU':
            loss = 'MSE'
        if args.qmode == 0: # NITI
            model = build_NITI_model(in_channel, img_dim, out_dim, args.model, 
                                     args.Wbitwidth, args.device,  Ab=args.Abitwidth, Eb=args.Ebitwidth, m=args.m, loss=loss)
        elif args.qmode == 1:
            model = build_q_model(in_channel, img_dim, out_dim, args.model, 
                                  args.Wbitwidth, args.batch_size, args.lr, args.device, Ab=args.Abitwidth, 
                                  Eb=args.Ebitwidth, stochastic=args.stochastic, loss=loss)
    return model

# def build_model(in_channel, img_dim, out_dim, args):
#     cfg = model_dict[args.model]
#     if args.qmode == 2:
#         model = nn_fp(in_channel, img_dim, out_dim, cfg, args.lr, args.momentum,args.use_bn)
#     else:
#         w_quant = lambda x: fp_quant(x, args.Wbitwidth - 1)
#         if args.stochastic:
#             e_quant = lambda x: fp_quant_stochastic(x, args.Ebitwidth - 1)
#             a_quant = lambda x: fp_quant_stochastic(x, args.Abitwidth - 1)
#         else:
#             e_quant = lambda x: fp_quant(x, args.Ebitwidth - 1)
#             a_quant = lambda x: fp_quant(x, args.Abitwidth - 1)

#         loss = QCELoss(e_quant)
#         if args.dataset == 'spectrum' or args.dataset == 'CWRU':
#             loss = QMSELoss(e_quant)
        
#         if args.qmode == 0: # NITI
#             weight_update = lambda a,b,c,d: NITI_weight_update(a,b,c,d,args.Wbitwidth - args.m, 2**(args.Wbitwidth-1)-1)
#             a_shit = lambda a, s: shift(a,s, args.Abitwidth - 1)
#             e_shift = lambda a, s : shift(a, s, args.Ebitwidth - 1)
#             model = nn_q(
#                 in_channel, img_dim, out_dim, cfg, loss, weight_update, a_shit, e_shift, 
#                      a_quant, w_quant, args.initialization,  use_bias=args.use_bn)
#         elif args.qmode == 1:
#             weight_update = lambda a, b, c, d: fp_weight_update(a,b,c,d,args.Wbitwidth - 1, args.batch_size, args.lr)
#             a_rescale = lambda a, s: a_quant(a*s[0])
#             e_rescale = lambda a, s: e_quant(a*s[0])
#             model = nn_q(in_channel, img_dim, out_dim, cfg, loss, weight_update, a_rescale, e_rescale, 
#                      a_quant, w_quant, args.initialization, use_bias=args.use_bn)


#     return model
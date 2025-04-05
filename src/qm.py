import torch 
from torch import nn
import torch.nn.functional as F
import collections

def combine_scale(s1, s2):
    return s1 * s2

class QLinear(nn.Module):
    def __init__(self, in_features, out_features, quantizer, weight_update, 
                 initialize='uniform',  bias=False):
        super(QLinear, self).__init__()

        # self.layer_init(in_features, out_features, qmode)
        self.quantizer = quantizer
        if bias:
            self.weight = torch.zeros([out_features, in_features+1])
        else:
            self.weight = torch.zeros([out_features, in_features])
        if initialize == 'uniform':
            torch.nn.init.xavier_uniform_(self.weight)
        elif initialize == 'normal':
            torch.nn.init.xavier_normal_(self.weight)
        
        self.weight, self.weight_scale= self.quantizer(self.weight)
        self.weight_update = weight_update
        self.bias = bias

    def forward(self, input):
        """ save activation for backwards """
        act, act_s = input

        if self.bias:
            act = torch.cat((act, torch.ones(act.shape[0], 1)), dim=1)
        self.act_in = act, act_s 

        out = torch.matmul(act, self.weight.T)

        out_s = combine_scale(act_s, self.weight_scale)
        return out, out_s
    

    def backward(self, input):
        # err: B x out_features
        err, err_s = input
        act, act_s = self.act_in

        self.grad = torch.matmul(err.T, act)
        self.grad_scale = combine_scale (err_s,  act_s)

        out = torch.matmul(err, self.weight)
        if self.bias:
            out = out[:, :-1]
        out_s = combine_scale(err_s,  self.weight_scale)
        self.weight, self.weight_scale = self.weight_update(self.weight, self.weight_scale, self.grad, self.grad_scale)

        return out, out_s
    
    def to(self, device):
        super().to(device)
        self.weight = self.weight.to(device)
        self.weight_scale = self.weight_scale.to(device)
        return self

class QConv2d(nn.Module):
    '''
    conv 3x3 with dilation 1, stride 1, padding 1
    NHWC format
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, 
                 quantizer, weight_update, initialize='uniform'):
        super(QConv2d, self).__init__()
        self.quantizer = quantizer

        self.weight = torch.zeros([out_channels, in_channels, kernel_size, kernel_size])
        if initialize == 'uniform':
            torch.nn.init.xavier_uniform_(self.weight)
        elif initialize == 'normal':
            torch.nn.init.xavier_normal_(self.weight)
        
        self.weight, self.weight_scale = self.quantizer(self.weight)
        self.weight_update = weight_update
        self.bias = False
        self.stride = stride
        self.padding = padding

    def forward(self, input):
        act, act_s = input
        self.act_in = act, act_s

        out = F.conv2d(act, self.weight, stride=self.stride, padding=self.padding)
        out_s = combine_scale(act_s, self.weight_scale)
        
        return out, out_s

    def backward(self, input):
        err, err_s = input

        act, act_s = self.act_in

        out = torch.nn.grad.conv2d_input(act.shape, self.weight, 
                                         err, self.stride, self.padding)
        self.grad = torch.nn.grad.conv2d_weight(act, self.weight.shape, err, self.stride, self.padding) 
        self.grad_scale = combine_scale(err_s, act_s)
        out_s = combine_scale(err_s, self.weight_scale)

        self.weight, self.weight_scale = self.weight_update(self.weight, self.weight_scale, self.grad, self.grad_scale)
        return out, out_s
    
    def to(self, device):
        super().to(device)
        self.weight = self.weight.to(device)
        self.weight_scale = self.weight_scale.to(device)
        return self
        

class QReLU(nn.Module):
    def __init__(self, forward_shift, backward_shift) -> None:
        super().__init__()
        self.forward_shift = forward_shift
        self.backward_shift = backward_shift


    def forward(self, input):
        self.act_in = input
        act, act_s = input
        out = torch.max(act, torch.tensor(0))
        out, out_s = self.forward_shift(out, act_s)
        return out, out_s

    def backward(self, input):
        err, err_s = input
        err, err_s = self.backward_shift(err, err_s)
        act, _ = self.act_in
        out = torch.where(act>0, err, torch.tensor(0))
        out_s = err_s
        return out, out_s

class QMaxpool2d(nn.Module):
    '''
    Integer Max Pooling 2d Layer
    '''
    def __init__(self, kernel_size, stride, padding=0):
        super(QMaxpool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self,input):
        act, act_s = input
        out, self.indices = F.max_pool2d(act, self.kernel_size, self.stride, self.padding, return_indices=True)
        out_s = act_s
        return out, out_s

    def backward(self, input):
        """ cudnn pool backward function doesn't support int8, use half instead"""
        err, err_s = input
        out = F.max_unpool2d(err.float(), self.indices, self.kernel_size, self.stride, self.padding)
        return out, err_s
    
class QGlobalPool2d(nn.Module):
    def __init__(self, forward_shift, backward_shift):
        super(QGlobalPool2d, self).__init__()
        # output_size can be an int or a tuple (H_out, W_out)
        self.output_size = (1,1)
        self.forward_shift = forward_shift
        self.backward_shift = backward_shift

    def forward(self, input):
        act, act_s = input
        self.input_shape = act.shape  # Save original shape (N, C, H, W)
        # Apply adaptive average pooling
        self.output = F.adaptive_avg_pool2d(act, self.output_size)  # shape: (N, C, H_out, W_out)
        return self.forward_shift(self.output, act_s)
    
    def backward(self, err):
        grad_out, grad_out_s = err
        N, C, H, W = self.input_shape

        # Create a tensor to hold the gradient for the input.
        region_area = H * W
        grad_act = grad_out.expand(N, C, H, W) 
        
        return self.backward_shift(grad_act, grad_out_s/region_area)
    
class QFlat(nn.Module):
    ''' Flat the input integer tensor except the batch dimension '''
    def forward(self, input):
        self.act_in = input
        act_in, exp_in = input
        return act_in.view(act_in.size(0), -1), exp_in

    def backward(self,input):
        '''
        Convert the Flat error back to the shape before flattern
        '''
        err_in, err_exp = input
        act, _ = self.act_in
        return err_in.view_as(act), err_exp


    
class QDropout(nn.Module):
    '''
    Integer Dropout layer
    '''
    def __init__(self, p=0.5,  *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.p = p
        self.training = True

    def forward(self, input):
        if self.training:
            act_in, exp_in = input
            self.drop_mask = torch.randint(low=0, high=2, size=(act_in.size(1),)).to(act_in.device)
            self.drop_mask = torch.where(self.drop_mask == 0, torch.tensor(0), torch.tensor(1))
            return act_in*self.drop_mask, exp_in
        return input

    def backward(self, input):
        err_in, err_exp = input
        err_out = err_in*self.drop_mask
        return err_out, err_exp


class QCELoss(nn.Module):
    def __init__(self, quantizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = quantizer

    def forward(self, out_val, scale, target):
        if torch.isnan(out_val).any() or torch.isinf(out_val).any():
            raise ValueError('out_val: nan or inf in CE loss')
        x = out_val * scale
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError('x: nan or inf in CE loss')
        y = F.softmax(x, dim=-1)
        if torch.isnan(y).any() or torch.isinf(y).any():
            raise ValueError('y: nan or inf in CE loss')
        x =  y - F.one_hot(target, out_val.size(1))
        return self.quantizer(x)   

class QMSELoss(nn.Module):
    def __init__(self, quantizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = quantizer

    def forward(self, out_val, scale, target):
        x =  (out_val * scale[0] - target) * 2
        return self.quantizer(x)  


class QBatchNorm2d(nn.Module):
    def __init__(self, num_features, weight_update, forward_rescale, backward_rescale ):
        super(QBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.weight = torch.ones(num_features)
        self.bias =  torch.zeros(num_features)
        self.forward_rescale = forward_rescale
        self.backward_rescale = backward_rescale
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.weight_update = weight_update

    def forward(self, input):
        # Unpack the input tuple: (activation, activation_scale)
        act, act_s = input
        self.input_shape = act.shape
        act = (act * act_s).view(-1, self.num_features) #(B*H*W, C)
        if self.training:
            # Compute batch statistics
            mean = act.mean(dim=0)  # shape (C,)
            var = act.var(dim=0, unbiased=False)  # shape (C,)
            with torch.no_grad():
                self.running_mean = 0.9* self.running_mean + 0.1 * mean
                self.running_var = 0.9 * self.running_var + 0.1 * act.var(dim=0, unbiased=True)
        else:
            # Use running statistics in evaluation mode
            mean = self.running_mean
            var = self.running_var

        # Normalize the activation
        inv_std = 1.0 / torch.sqrt(var + 1e-5)
        norm_act = (act - mean) * inv_std
        out = self.weight * norm_act + self.bias
        self.cache = (act, norm_act, inv_std)

        return self.forward_rescale(out.view(self.input_shape), 1)
    
    def backward(self, input):
        err, err_s = input

        err = err.view(-1, self.num_features)
        N = err.shape[0]

        act, norm_act, inv_std = self.cache

        dgamma = (err * norm_act).sum(dim=0)  # shape (C,)
        dbeta = err.sum(dim=0)                # shape (C,)
        # For the backward pass through the affine layer:
        dnorm_act = err * self.weight * err_s # shape (N, C)
        sum_dnorm = dnorm_act.sum(dim=0)               # shape (C,)
        sum_dnorm_norm = (dnorm_act * norm_act).sum(dim=0)  # shape (C,)
        dact = (1.0 / N) * inv_std * (
                    N * dnorm_act - sum_dnorm[None, :] - norm_act * sum_dnorm_norm[None, :]
                )  # shape (N, C)
        
        grad_act = dact.view(self.input_shape)  # Reshape to original (N, C, H, W)

        self.weight, w1 = self.weight_update(self.weight, 1, dgamma, err_s)
        self.bias, b1 = self.weight_update(self.bias, 1, dbeta, err_s)
        self.weight *= w1
        self.bias *= b1
    
        return self.backward_rescale(grad_act, 1)
    def to(self, device):
        super().to(device)
        self.weight = self.weight.to(device)
        self.bias = self.bias.to(device)
        self.running_mean = self.running_mean.to(device)
        self.running_var = self.running_var.to(device)
        return self
    

def _collect_state_dict(module, prefix=''):
    state = collections.OrderedDict()
    # If this module is a "leaf" module with custom state:
    if hasattr(module, 'weight'):
        if hasattr(module, 'weight_scale'):
            state[prefix + 'weight'] = module.weight
            state[prefix + 'weight_scale'] = module.weight_scale
        else:
            state[prefix + 'weight'] = module.weight
            if hasattr(module, 'bias'):
                state[prefix + 'bias'] = module.bias
            if hasattr(module, 'running_mean'):
                state[prefix + 'running_mean'] = module.running_mean
            if hasattr(module, 'running_var'):
                state[prefix + 'running_var'] = module.running_var
    # Recurse into children.
    for name, child in module.named_children():
        child_prefix = prefix + name + '.'
        state.update(_collect_state_dict(child, child_prefix))
    return state

def _load_state_dict(module, state, prefix=''):
    """
    Recursively loads state into a module and its children.
    It looks for keys with the given prefix and assigns them to the appropriate attributes.
    """
    if hasattr(module, 'weight'):
        if hasattr(module, 'weight_scale'):
            module.weight = state[prefix + 'weight']
            module.weight, module.weight_scale = module.quantizer(module.weight)
        else:
            module.weight = state[prefix + 'weight']
            if prefix + 'bias' in state:
                module.bias = state[prefix + 'bias']
            if prefix + 'running_mean' in state:
                module.running_mean = state[prefix + 'running_mean']
            if prefix + 'running_var' in state:
                module.running_var = state[prefix + 'running_var']
    # Recurse into children.
    for name, child in module.named_children():
        child_prefix = prefix + name + '.'
        _load_state_dict(child, state, child_prefix)
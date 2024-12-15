import torch
import torch.nn.functional as F


#NITI

def RoundShift(input, shift):
    '''
    Shift the input using
    stochastic rounding
    '''
    round_temp = input//(2**shift)
    prob = input - round_temp * (2**shift)
    round_decision = prob//(2**(shift-1))
    return round_temp + round_decision

def StoShift(input, shift):
    '''
    Shift the input using
    stochastic rounding
    '''
    tensor_type = input.dtype
    round_temp = input//(2**shift)
    if shift <= 0:
        return round_temp
    prob = torch.abs(input - round_temp * (2**shift))
    rand_num = torch.randint(low = 0, high=2**shift,size=prob.size(), dtype = tensor_type).to(input.device)
    round_decision = torch.where(prob <= rand_num,
                                 torch.tensor(0,dtype=tensor_type),
                                 torch.tensor(1,dtype=tensor_type))
    round_decision = round_decision * torch.sign(input - round_temp * (2**shift))
    return round_temp + round_decision

def get_bitwidth(input):
    input_range = torch.max(torch.abs(input))
    if input_range == 0:
        return 0
    return torch.ceil(torch.log2(input_range)).int()

def fp_quant_2log(input, bitwidth):
    input_range = torch.max(torch.abs(input))
    input_bitwidth=torch.ceil(torch.log2(input_range))
    act_exp = input_bitwidth - bitwidth
    round_val = torch.round(input/input_range*(2**bitwidth-1))
    return round_val, [2**act_exp]

def int8_quant(input):
    return fp_quant_2log(input, 7)

def uint8_quant(input):
    return fp_quant_2log(input, 8)

def shift(input, s, target_bitwidth): 
    bw = get_bitwidth(input)
    range = 2**target_bitwidth - 1
    if bw > target_bitwidth:
        return int8_clip(RoundShift(input, bw-target_bitwidth), range), [2**(bw - target_bitwidth) * s[0]]
    return input, s

    
def int8_clip(input, clip_val=127):
    return torch.clamp(input,-clip_val, clip_val)

def fp_quant(input, bitwidth):
    input_range = torch.max(torch.abs(input))
    unround = input/input_range*(2**bitwidth - 1)
    return deterministic_round(unround), [input_range/(2**bitwidth - 1)]

def fp_quant_stochastic(input, bitwidth):
    if torch.isnan(input).any():
        raise ValueError('fp_quant NaN in input')
    
    input_range = torch.max(torch.abs(input))
    unround = input/input_range*(2**bitwidth - 1)

    return stochastic_round(unround), [input_range/(2**bitwidth - 1)]
    
def stochastic_round(t):
    if torch.isnan(t).any():
        raise ValueError('stochastic NaN in unround')
    floor = torch.floor(t)
    ceil = torch.ceil(t)
    return torch.where(torch.rand_like(t) > (t - floor), floor, ceil)

def deterministic_round(t):
    floor = torch.floor(t)
    ceil = torch.ceil(t)
    return torch.where(t - floor<0.5, floor, ceil)

def cov_backward(X, w, E, padding=1, stride=1):
    PX = F.pad(X, (padding, padding, padding, padding))
    N, C, H, W = X.shape
    M, _, FH, FW = w.shape

    _, _, padded_H, padded_W = PX.shape
    dx = torch.zeros_like(X)
    dW = torch.zeros_like(w)

    x_col = torch.zeros((C * FH * FW, H * W))
    w_row = w.reshape(M, C * FH * FW)

    for i in range(N):
        curr_dout = E[i, :, :, :].reshape(M, H * W)
        curr_out = w_row.T @  curr_dout
        curr_dpx = torch.zeros(PX.shape[1:])
        c = 0
        for j in range(0, padded_H - FH + 1, stride):
            for k in range(0, padded_W - FW + 1, stride):
                curr_dpx[:, j:j+FH, k:k+FW] += curr_out[:, c].reshape(C, FH, FW)
                x_col[:, c] = PX[i, :, j:j+FH, k:k+FW].reshape(C * FH * FW) 
                c += 1
        dx[i] = curr_dpx[:, padding:-padding, padding:-padding]
        dW += (curr_dout @ x_col.T).reshape(M, C, FH, FW)
    return dW, dx
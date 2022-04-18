import torch
from utils.batch_interpolate import batch_interp

def initial_condition_loss(inputs, model):
    grid_size = model.grid_size

    inputs_initial = inputs.clone()
    inputs_initial[:,1] = 0
    outputs_initial = model(inputs_initial)
    
    psi0_real = inputs[:,2:grid_size+2]
    psi0_imag = inputs[:,grid_size+2:2*grid_size+2]
    xs = inputs[:,0]
    
    targets_real = batch_interp(psi0_real, xs)
    targets_imag = batch_interp(psi0_imag, xs)
    
    return torch.mean((outputs_initial[:,0] - targets_real[:,0])**2 + (outputs_initial[:,1] - targets_imag[:,0])**2)

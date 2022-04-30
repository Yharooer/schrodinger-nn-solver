import torch
from ..batch_interpolate import batch_eval

def differential_loss(output, inputs, model, use_autograd=False):
    batch_size = output.shape[0]
    
    psi_real = output[:,0]
    psi_imag = output[:,1]
    
    # Calculate Laplacian
    if use_autograd:
        e_x = torch.zeros((batch_size,302)).to(device)
        e_x[:,0] = 1
        psi_dx = lambda i: torch.autograd.functional.jvp(model, i, v=e_x, create_graph=True)[1]
        psi_d2x = torch.autograd.functional.jvp(psi_dx, inputs, v=e_x, create_graph=True)[1]
    else:
        dx = 1e-2
        inputs_x_minus = inputs.clone()
        inputs_x_minus[:,0] -= dx
        inputs_x_plus = inputs.clone()
        inputs_x_plus[:,0] += dx

        psi_x_minus = model(inputs_x_minus)
        psi_x_plus = model(inputs_x_plus)
        psi_d2x = (psi_x_plus + psi_x_minus - 2*output)/(dx**2)
    
    psi_d2x_real = psi_d2x[:,0]
    psi_d2x_imag = psi_d2x[:,1]
    
    # Calculate time derivative
    if use_autograd:
        e_t = torch.zeros((batch_size,302)).to(device)
        e_t[:,1] = 1
        psi_dt = torch.autograd.functional.jvp(model, inputs, v=e_t, create_graph=True)[1]
    else:
        dt = 1e-5
        inputs_t_minus = inputs.clone()
        inputs_t_minus[:,1] -= dt
        inputs_t_plus = inputs.clone()
        inputs_t_plus[:,1] += dt

        psi_t_minus = model(inputs_t_minus)
        psi_t_plus = model(inputs_t_plus)
        psi_dt = (psi_t_plus - psi_t_minus)/(2*dt)
    
    psi_dt_real = psi_dt[:,0]
    psi_dt_imag = psi_dt[:,1]
    
    # Calculate potential energy
    V_grid = inputs[:,202:]
    V = batch_eval(V_grid, inputs[:,0])

    V_real = V[:,0] * psi_real
    V_imag = V[:,0] * psi_imag

    # Calculate loss
    diff_1 = psi_dt_real - 0.5*psi_d2x_imag + V_imag
    diff_2 = psi_dt_imag + 0.5*psi_d2x_real - V_real

    return torch.mean(diff_1**2 + diff_2**2)
import torch
from ..batch_interpolate import batch_eval
from ..legendre_coefficients import get_legendre_coefficients

# import matplotlib.pyplot as plt
# import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
legendre_n, legendre_xs, legendre_ws = get_legendre_coefficients(n=7)

def energy_conservation_loss(inputs, model, use_autograd=False):
    # Reshape input to evalute at legendre_n number of points.
    batch_size = inputs.shape[0]
    inputs_clone = inputs.clone()
    inputs_clone = inputs_clone.expand(legendre_n,-1,-1).clone()
    xs_expanded = legendre_xs.expand(batch_size, -1).T
    inputs_clone[:,:,0] = xs_expanded
    inputs_clone = torch.reshape(inputs_clone, (batch_size*legendre_n, -1))

    psi = model(inputs_clone)
    psi = torch.reshape(psi, (legendre_n, batch_size, -1))

    normalisation = torch.inner((psi[:,:,0]**2 + psi[:,:,1]**2).T, legendre_ws)/2

    if use_autograd:
        e_x = torch.zeros((inputs_clone.shape[0],302)).to(device)
        e_x[:,0] = 1
        psi_dx = torch.autograd.functional.jvp(model, inputs_clone, v=e_x, create_graph=True)[1]
    else:
        dx = 1e-5
        inputs_x_minus = inputs_clone.clone()
        inputs_x_minus[:,0] -= dx
        inputs_x_plus = inputs_clone.clone()
        inputs_x_plus[:,0] += dx

        psi_x_minus = model(inputs_x_minus)
        psi_x_plus = model(inputs_x_plus)
        psi_dx = (psi_x_plus - psi_x_minus)/(2*dx)

    psi_dx = torch.reshape(psi_dx, (legendre_n, batch_size, -1))

    # Find potential term
    V_grid = inputs[:,202:]
    V_grid_expanded = V_grid.expand(legendre_n, -1, -1).clone()
    V_grid_reshaped = torch.reshape(V_grid_expanded, (batch_size*legendre_n, -1))

    legendre_xs_reshaped = torch.reshape(xs_expanded, (batch_size*legendre_n,))

    V_sampled_reshaped = batch_eval(V_grid_reshaped, legendre_xs_reshaped)
    V_sampled = torch.reshape(V_sampled_reshaped, (legendre_n, batch_size))

    # plt.figure()
    # plt.plot(np.linspace(0,1,100), V_grid[0,:].detach().cpu().numpy())
    # plt.plot(legendre_xs.cpu().numpy(), V_sampled[:,0].detach().cpu().numpy())
    # plt.show()

    # Now use <H> = int 0.5*(psi_dx) + V(x)[psi^2].
    psi_h_psi = 0.5*(psi_dx[:,:,0]**2 + psi_dx[:,:,1]**2) + V_sampled*(psi[:,:,0]**2 + psi[:,:,1]**2)
    expectation_energy_after = torch.inner(psi_h_psi.T, legendre_ws)/2/normalisation

    # Calculate initial energy expectation
    psi_0_real = inputs[:,2:102]
    psi_0_imag = inputs[:,102:202]
    psi_0_real_dx = (psi_0_real[:,1:] - psi_0_real[:,:-1])*100 
    psi_0_imag_dx = (psi_0_imag[:,1:] - psi_0_imag[:,:-1])*100

    initial_kinetic = torch.sum(0.5*(psi_0_real_dx**2 + psi_0_imag_dx**2), dim=1)/99
    
    v = inputs[:,202:]
    initial_potential = torch.sum(v*(psi_0_real**2 + psi_0_imag**2), dim=1)/100

    initial_expectation = initial_kinetic + initial_potential

    return torch.mean((1 - expectation_energy_after / initial_expectation)**2)

import torch
from ..legendre_coefficients import get_legendre_coefficients

legendre_n, legendre_xs, legendre_ws = get_legendre_coefficients(n=7)

def normalisation_loss(inputs, model):
    batch_size = inputs.shape[0]
    inputs_clone = inputs.clone()
    inputs_clone = inputs_clone.expand(legendre_n,-1,-1).clone()
    xs_expanded = legendre_xs.expand(batch_size, -1).T
    inputs_clone[:,:,0] = xs_expanded
    inputs_clone = torch.reshape(inputs_clone, (batch_size*legendre_n, -1))
    outputs = model(inputs_clone)
    outputs = torch.reshape(outputs, (legendre_n, batch_size, -1))
    outputs = outputs[:,:,0]**2 + outputs[:,:,1]**2
    norms = torch.inner(outputs.T, legendre_ws)/2
    return torch.mean((norms-1)**2)
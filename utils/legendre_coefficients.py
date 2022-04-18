import torch
import scipy.special

# TODO warning this will fail if later we set multiple gpu
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_legendre_coefficients(n=5):
    legendre_n = n
    legendre_xs, legendre_ws = scipy.special.roots_legendre(legendre_n)
    legendre_xs = (legendre_xs + 1.0)/2.0
    legendre_xs = torch.tensor(legendre_xs).to(device).float()
    legendre_ws = torch.tensor(legendre_ws).to(device).float()

    return legendre_n, legendre_xs, legendre_ws
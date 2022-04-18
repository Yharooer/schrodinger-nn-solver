from .loss_functions.differential_loss import differential_loss
from .loss_functions.boundary_condition_loss import boundary_condition_loss
from .loss_functions.initial_condition_loss import initial_condition_loss
from .loss_functions.normalisation_loss import normalisation_loss
from .loss_functions.energy_conservation_loss import energy_conservation_loss
import torch.nn.functional as F


def get_loss_full_labels():
    return ['MSE', 'Differential', 'Boundary Conditions', 'Initial Conditions', 'Normalisation', 'Energy Conservation']


def get_loss_short_labels():
    return ['MSE', 'dt', 'BCs', 'ICs', 'Norm.', 'Energy']


def loss_function(outputs, inputs, targets, model, HYPERPARAM_MSE, HYPERPARAM_DT, HYPERPARAM_BC, HYPERPARAM_IC, HYPERPARAM_NORM, HYPERPARAM_ENERGY, use_autograd=False, lazy_eval=True):
    loss_mse = F.mse_loss(outputs, targets)
    loss_dt = differential_loss(outputs, inputs, model, use_autograd) if HYPERPARAM_DT != 0 or (not lazy_eval) else 0
    loss_bc = boundary_condition_loss(inputs, model) if HYPERPARAM_BC != 0 or (not lazy_eval) else 0
    loss_ic = initial_condition_loss(inputs, model) if HYPERPARAM_IC != 0 or (not lazy_eval) else 0
    loss_norm = normalisation_loss(inputs, model) if HYPERPARAM_NORM != 0 or (not lazy_eval) else 0
    loss_energy = energy_conservation_loss(inputs, model, use_autograd) if HYPERPARAM_ENERGY != 0 or (not lazy_eval) else 0

    loss_total = (
        HYPERPARAM_MSE * loss_mse +
        HYPERPARAM_DT * loss_dt +
        HYPERPARAM_BC * loss_bc +
        HYPERPARAM_IC * loss_ic +
        HYPERPARAM_NORM * loss_norm +
        HYPERPARAM_ENERGY * loss_energy
    )

    return loss_total, [loss_mse, loss_dt, loss_bc, loss_ic, loss_norm, loss_energy]
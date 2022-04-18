import torch

def boundary_condition_loss(inputs, model):    
    inputs_left = inputs.clone()
    inputs_left[:,0] = 0
    outputs_left = model(inputs_left)
    
    inputs_right = inputs.clone()
    inputs_right[:,0] = 1
    outputs_right = model(inputs_right)
    
    return torch.mean(outputs_left**2 + outputs_right**2)
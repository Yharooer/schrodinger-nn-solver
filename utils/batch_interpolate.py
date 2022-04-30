import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Evaluates batch_size number of functions on [0,1] each at a position x
# func: shape   (batch_size, grid_length)
# x   : shape   (batch_size,)
# returns shape (batch_size, 1)
def batch_eval(funcs, x):
    batch_size = funcs.shape[0]
    grid_length = funcs.shape[1]
    
    left_pos = torch.zeros((batch_size,1)).type(torch.LongTensor).to(device)
    left_pos[:,0] = torch.floor(x*grid_length)
    
    right_pos = left_pos.clone()
    right_pos += 1
    
    min_pos = torch.zeros_like(left_pos)
    max_pos = torch.zeros_like(left_pos)
    max_pos += grid_length - 1
    
    right_pos = torch.maximum(torch.minimum(right_pos, max_pos), min_pos)
    left_pos = torch.maximum(torch.minimum(left_pos, max_pos), min_pos)
    
    s = torch.zeros((batch_size,1)).to(device)
    s[:,0] = x*grid_length - torch.floor(x*grid_length)
    
    left_sample = torch.gather(funcs, 1, left_pos)
    right_sample = torch.gather(funcs, 1, right_pos)
    
    return (1-s)*left_sample + s*right_sample

# Evaluates batch_size number of functions [0,1] each at positions xs
# func: shape   (batch_size, grid_length)
# xs  : shape   (batch_size, xs_num)
# returns shape (batch_size, xs_num)
# TODO can refactor energy_conservation_loss to use this.
def batch_interp(func, xs):
    batch_size = func.shape[0]
    grid_length = func.shape[1]
    xs_num = xs.shape[1]

    if xs.shape[0] != batch_size:
        print(f'Shape of func is {func.shape}.')
        print(f'Shape of xs is {xs.shape}.')
        raise ValueError('xs.shape[0] and func.shape[0] must be the same and should be the batch number.')

    func = func.expand(xs_num,-1,-1).clone()
    func = torch.reshape(func, (xs_num*batch_size, grid_length))

    xs = torch.reshape(xs.T, (xs_num*batch_size,))

    result = batch_eval(func, xs)

    return torch.reshape(result, (xs_num, batch_size)).T
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def batch_interp(funcs, xs):
    batch_size = funcs.shape[0]
    grid_length = funcs.shape[1]
    
    left_pos = torch.zeros((batch_size,1)).type(torch.LongTensor).to(device)
    left_pos[:,0] = torch.floor(xs*grid_length)
    
    right_pos = left_pos.clone()
    right_pos += 1
    
    min_pos = torch.zeros_like(left_pos)
    max_pos = torch.zeros_like(left_pos)
    max_pos += grid_length - 1
    
    right_pos = torch.maximum(torch.minimum(right_pos, max_pos), min_pos)
    left_pos = torch.maximum(torch.minimum(left_pos, max_pos), min_pos)
    
    s = torch.zeros((batch_size,1)).to(device)
    s[:,0] = xs*grid_length - torch.floor(xs*grid_length)
    
    left_sample = torch.gather(funcs, 1, left_pos)
    right_sample = torch.gather(funcs, 1, right_pos)
    
    return (1-s)*left_sample + s*right_sample
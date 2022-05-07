import torch.nn as nn

class SchrodingerModel(nn.Module):
    def __init__(self, grid_size=100, hidden_dim=300, num_layers=2):
        super(SchrodingerModel, self).__init__()
        
        self.grid_size = grid_size

        sequentials = [nn.Linear(3*grid_size+2, hidden_dim)]
        for i in range(num_layers):
            sequentials.append(nn.Softplus())
            sequentials.append(nn.Linear(hidden_dim, hidden_dim))
        sequentials.append(nn.Softplus())
        sequentials.append(nn.Linear(hidden_dim, 2))

        self.mlp = nn.Sequential(*sequentials)

    def forward(self, x):
        return self.mlp(x)
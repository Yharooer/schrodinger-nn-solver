import torch.nn as nn

class SchrodingerModel(nn.Module):
    def __init__(self, grid_size=100, hidden_dim=300):
        super(SchrodingerModel, self).__init__()
        
        self.grid_size = grid_size

        self.mlp = nn.Sequential(
            nn.Linear(3*grid_size+2, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.mlp(x)
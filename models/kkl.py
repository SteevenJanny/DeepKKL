import torch
import torch.nn as nn


class KKL(nn.Module):
    def __init__(self, observation_dim, latent_space_dim, prediction_horizon, mlp_hidden_dim):
        super(KKL, self).__init__()
        self.latent_dim = latent_space_dim
        self.horizon = prediction_horizon

        self.A = nn.Parameter(torch.randn(latent_space_dim, latent_space_dim) / latent_space_dim, requires_grad=True)
        self.b = nn.Parameter(torch.ones(latent_space_dim), requires_grad=False)

        self.psi = nn.Sequential(
            nn.Linear(latent_space_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, observation_dim)
        )

    def forward(self, y):
        B, T, S = y.shape
        y_hat = []
        z = torch.zeros(B, self.latent_dim)

        # Contraction
        for t in range(T):
            y_hat.append(self.psi(z))
            z = self.dynamic(z, y[:, t])

        # Prediction
        for t in range(T, self.horizon):
            y_hat.append(self.psi(z))
            z = self.dynamic(z, y_hat[-1])

        y_hat = torch.stack(y_hat, dim=1)
        return y_hat

    def dynamic(self, z, y):
        B, S = z.shape
        A = self.A.unsqueeze(0).repeat(B, 1, 1)
        B = self.b.unsqueeze(0).repeat(B, 1).unsqueeze(-1)
        dz = (A @ z.unsqueeze(-1) + B @ y.unsqueeze(-1)).squeeze(-1)
        return dz
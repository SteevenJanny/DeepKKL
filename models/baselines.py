import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, observation_dim, latent_space_dim, prediction_horizon, mlp_hidden_dim):
        super(RNN, self).__init__()
        self.latent_dim = latent_space_dim
        self.horizon = prediction_horizon

        self.dynamic = nn.RNN(input_size=observation_dim, hidden_size=latent_space_dim, batch_first=True)

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
        z = torch.zeros(1, B, self.latent_dim)

        # Contraction
        for t in range(T):
            y_hat.append(self.psi(z).view(B, -1))
            _, z = self.dynamic(y[:, t].unsqueeze(1), z)

        # Prediction
        for t in range(T, self.horizon):
            y_hat.append(self.psi(z).view(B, -1))
            _, z = self.dynamic(y_hat[-1].unsqueeze(1), z)

        y_hat = torch.stack(y_hat, dim=1)
        return y_hat


class GRU(nn.Module):
    def __init__(self, observation_dim, latent_space_dim, prediction_horizon, mlp_hidden_dim):
        super(GRU, self).__init__()
        self.latent_dim = latent_space_dim
        self.horizon = prediction_horizon

        self.dynamic = nn.GRU(input_size=observation_dim, hidden_size=latent_space_dim, batch_first=True)

        self.psi = nn.Sequential(
            nn.Linear(latent_space_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, observation_dim)
        )

    def forward(self, y):
        B, T, S = y.shape
        y_hat = []
        z = torch.zeros(1, B, self.latent_dim)

        # Contraction
        for t in range(T):
            y_hat.append(self.psi(z).view(B, -1))
            _, z = self.dynamic(y[:, t].unsqueeze(1), z)

        # Prediction
        for t in range(T, self.horizon):
            y_hat.append(self.psi(z).view(B, -1))
            _, z = self.dynamic(y_hat[-1].unsqueeze(1), z)

        y_hat = torch.stack(y_hat, dim=1)
        return y_hat

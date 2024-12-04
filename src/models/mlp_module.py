import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPProjection(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_output):
        # only use first token ([CLS]) of each output
        x = x_output[:, 0, :]

        x = self.linear1(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x


class MLPPrediction(nn.Module):
    def __init__(self, input_dim, use_abs_diff=False, use_mult=False):
        super().__init__()

        self.use_abs_diff = use_abs_diff
        self.use_mult = use_mult

        real_input_dim = input_dim * (2 + int(use_abs_diff) + int(use_mult))

        self.mlp = nn.Sequential(
            nn.Linear(real_input_dim, 512),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)

        if self.use_abs_diff:
            x_diff = torch.abs(x1 - x2)
            x = torch.cat([x, x_diff], dim=1)

        if self.use_mult:
            x_mult = x1 * x2
            x = torch.cat([x, x_mult], dim=1)

        x = self.mlp(x)

        return x

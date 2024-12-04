import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnProjection(nn.Module):

    def __init__(self, input_dim, n_heads, output_length):
        super().__init__()

        self.query = nn.Parameter(torch.randn(output_length, input_dim))

        self.attn = nn.MultiheadAttention(
            input_dim, n_heads, dropout=0.2, batch_first=True
        )
        self.norm1 = nn.LayerNorm(input_dim)

        self.self_attn = nn.MultiheadAttention(
            input_dim, n_heads, dropout=0.2, batch_first=True
        )
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.2)

        nn.init.xavier_normal_(self.query)

    def forward(self, x):
        B = x.shape[0]

        query = self.query.unsqueeze(0).repeat(B, 1, 1)

        z = self.norm1(x)
        z_attn = self.attn(query, z, z)[0]
        z = z_attn

        z = self.norm2(z)
        z_attn = self.self_attn(z, z, z)[0]
        z = z + self.dropout(z_attn)

        z = z.contiguous().view(B, -1)

        return z


class BiAttnPrediction(nn.Module):

    def __init__(self, input_dim, n_heads):
        super().__init__()

        self.input_dim = input_dim

        self.attn1 = nn.MultiheadAttention(
            input_dim, n_heads, dropout=0.2, batch_first=True
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(0.2)

        self.attn2 = nn.MultiheadAttention(
            input_dim, n_heads, dropout=0.2, batch_first=True
        )
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout2 = nn.Dropout(0.2)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 3, 1024),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x1, x2):
        B = x1.shape[0]
        x1 = x1.view(B, -1, self.input_dim)  # [B, M x D] -> [B, M, D]
        x2 = x2.view(B, -1, self.input_dim)  # [B, M x D] -> [B, M, D]

        x1 = self.norm1(x1)
        x2 = self.norm2(x2)

        z1_attn = self.attn1(x2, x1, x1)[0]
        z1 = x1 + self.dropout1(z1_attn)

        z2_attn = self.attn2(x1, x2, x2)[0]
        z2 = x2 + self.dropout2(z2_attn)

        z1 = z1.mean(dim=1)
        z2 = z2.mean(dim=1)

        z = torch.cat([z1, z2, torch.abs(z1 - z2)], dim=1)  # [B, D * 4]

        z = self.mlp(z)

        return z

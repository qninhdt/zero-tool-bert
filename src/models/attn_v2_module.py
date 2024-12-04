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
        self.dropout1 = nn.Dropout(0.2)

        self.self_attn = nn.MultiheadAttention(
            input_dim, n_heads, dropout=0.2, batch_first=True
        )
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout2 = nn.Dropout(0.2)

        self.cls_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.SiLU(), nn.Dropout(0.2)
        )
        self.norm3 = nn.LayerNorm(input_dim)

        nn.init.xavier_normal_(self.query)

    def forward(self, x):
        B = x.shape[0]

        query = self.query.unsqueeze(0).repeat(B, 1, 1)

        x_cls = x[:, 0, :]
        x_other = x[:, 1:, :]

        z_other = self.norm1(x_other)
        z_attn = self.attn(query, z_other, z_other)[0]
        z_other = self.dropout1(z_attn)

        z_other = self.norm2(z_other)
        z_attn = self.self_attn(z_other, z_other, z_other)[0]
        z_other = z_other + self.dropout1(z_attn)

        z_cls = x_cls + self.cls_mlp(self.norm3(x_cls))

        z = torch.cat([z_cls.unsqueeze(1), z_other], dim=1)

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
            nn.Linear(input_dim * 6, 1024),
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
        self.norm3 = nn.LayerNorm(input_dim)

    def forward(self, x1, x2):
        B = x1.shape[0]
        x1 = x1.view(B, -1, self.input_dim)  # [B, M x D] -> [B, M, D]
        x2 = x2.view(B, -1, self.input_dim)  # [B, M x D] -> [B, M, D]

        z1_cls = x1[:, 0, :]
        z2_cls = x2[:, 0, :]

        x1_other = self.norm1(x1[:, 1:, :])
        x2_other = self.norm2(x2[:, 1:, :])

        z1_attn = self.attn1(x2_other, x1_other, x1_other)[0]
        z1_other = x1_other + self.dropout1(z1_attn)

        z2_attn = self.attn2(x1_other, x2_other, x2_other)[0]
        z2_other = x2_other + self.dropout2(z2_attn)

        z1_other = z1_other.mean(dim=1)
        z2_other = z2_other.mean(dim=1)

        z = torch.cat(
            [
                z1_cls,
                z1_other,
                z2_cls,
                z2_other,
                torch.abs(z1_cls - z2_cls),
                torch.abs(z1_other - z2_other),
            ],
            dim=1,
        )  # [B, D * 4]

        z = self.mlp(z)

        return z

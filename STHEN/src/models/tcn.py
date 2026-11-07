

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int,
                 kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               padding=0, dilation=dilation)
        self.pad = nn.ConstantPad1d((padding, 0), 0.0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.norm = nn.GroupNorm(1, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pad(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.norm(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNChannel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_levels: int = 4, kernel_size: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        layers = []
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_dim if i == 0 else hidden_dim
            layers.append(TemporalBlock(in_ch, hidden_dim, kernel_size,
                                        dilation, dropout))

        self.tcn_blocks = nn.Sequential(*layers)


        self.tcp_head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):

        x = x.permute(0, 2, 1)

        tcn_feat = self.tcn_blocks(x)   

        H_trend, _ = tcn_feat.max(dim=2)  

        tcp_score = self.tcp_head(tcn_feat).squeeze(1)  

        return H_trend, tcp_score, tcn_feat



import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicAttentionFusion(nn.Module):


    def __init__(self, d_time: int, d_space: int, d_trend: int,
                 d_model: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.proj_time  = nn.Linear(d_time,  d_model)
        self.proj_space = nn.Linear(d_space, d_model)
        self.proj_trend = nn.Linear(d_trend, d_model)

        self.attn = nn.MultiheadAttention(embed_dim=d_model,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          batch_first=True)

     model * 3, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, H_time: torch.Tensor, H_space: torch.Tensor,
                H_trend: torch.Tensor):

        N = H_time.shape[0]

        Q_time  = self.proj_time(H_time)    
        Q_space = self.proj_space(H_space)
        Q_trend = self.proj_trend(H_trend)

        features = torch.stack([Q_time, Q_space, Q_trend], dim=1)

        attn_out, attn_w = self.attn(features, features, features,
                                      need_weights=True, average_attn_weights=False)
      
        fused = attn_out.reshape(N, -1)    
        H_fused = self.fc_out(fused)      
        H_fused = self.norm(H_fused)

        return H_fused, attn_w


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class TopoVeloGAT(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.1, concat: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.concat = concat

        layers = []
        skip_projs = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else hidden_dim
            is_last = (i == num_layers - 1)

            layers.append(GATConv(
                in_channels=in_dim,
                out_channels=out_dim,
                heads=num_heads,
                dropout=dropout,
                concat=concat,
                add_self_loops=False
            ))

            gat_out_dim = out_dim * num_heads if concat else out_dim
            skip_projs.append(
                nn.Linear(in_dim, gat_out_dim) if in_dim != gat_out_dim else nn.Identity()
            )

            in_dim = out_dim * num_heads if concat else out_dim

        self.gat_layers = nn.ModuleList(layers)
        self.skip_projs = nn.ModuleList(skip_projs)
        final_dim = hidden_dim * num_heads if concat else hidden_dim
        self.norm = nn.LayerNorm(final_dim)


        self.scm_head = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x_last: torch.Tensor, edge_index: torch.Tensor):
 
        num_nodes = x_last.shape[0]
        edge_max = edge_index.max().item() if edge_index.numel() > 0 else -1
        edge_min = edge_index.min().item() if edge_index.numel() > 0 else 0
        assert edge_min >= 0, f"edge_index has negative index: {edge_min}"
        assert edge_max < num_nodes, \
            f"edge_index out of bounds: max={edge_max}, num_nodes={num_nodes}"

        att_weights_list = []
        for i, gat_layer in enumerate(self.gat_layers):
            residual = self.skip_projs[i](x_last)
            x_last, (edge_idx, att_w) = gat_layer(x_last, edge_index,
                                                    return_attention_weights=True)
            att_weights_list.append(att_w)
            x_last = x_last + residual

            if i < len(self.gat_layers) - 1:
                x_last = F.elu(x_last)

        H_space = self.norm(x_last)

        influence_scores = self._compute_influence_score(
            edge_index, att_weights_list[-1], H_space.shape[0]
        )

        scm_score = self.scm_head(H_space).squeeze(-1)

        return H_space, influence_scores, scm_score

    def _compute_influence_score(self, edge_index: torch.Tensor,
                                  att_weights: torch.Tensor,
                                  num_nodes: int) -> torch.Tensor:
        target_nodes = edge_index[1]
        N = num_nodes

        att_avg = att_weights.mean(dim=1)

        influence = torch.zeros(N, device=edge_index.device)
        influence.index_add_(0, target_nodes, att_avg)

        return influence

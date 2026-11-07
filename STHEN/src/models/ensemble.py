
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerClassifier(nn.Module):


    def __init__(self, d_model: int, num_layers: int = 2,
                 num_heads: int = 4, num_classes: int = 6,
                 dim_feedforward: int = None, dropout: float = 0.1):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout),
                nn.LayerNorm(d_model)
            ])
        self.encoder = nn.Sequential(*layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.encoder(x)   
        return self.fc(x)     


class TCNClassifier(nn.Module):


    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 3, num_classes: int = 6,
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()

        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class OdeLSTMClassifier(nn.Module):


    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, num_classes: int = 6, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class HeterogeneousEnsemble(nn.Module):


    def __init__(self, d_fused: int = 256, num_classes: int = 6,
                 learnable_weights: bool = True,
                 transformer_cfg: dict = None,
                 tcn_cfg: dict = None,
                 ode_cfg: dict = None):
        super().__init__()


        transformer_cfg = transformer_cfg or {'num_layers': 2, 'num_heads': 4}
        tcn_cfg = tcn_cfg or {'num_layers': 3}
        ode_cfg = ode_cfg or {'num_layers': 2}


        self.clf_transformer = TransformerClassifier(
            d_model=d_fused, num_classes=num_classes, **transformer_cfg
        )
        self.clf_tcn = TCNClassifier(
            input_dim=d_fused, num_classes=num_classes, **tcn_cfg
        )
        self.clf_ode = OdeLSTMClassifier(
            input_dim=d_fused, num_classes=num_classes, **ode_cfg
        )


        if learnable_weights:
            self.ensemble_weights = nn.Parameter(torch.ones(3) / 3.0)
        else:
            self.register_buffer('ensemble_weights', torch.ones(3) / 3.0)

    def forward(self, H_fused: torch.Tensor):

        logits_1 = self.clf_transformer(H_fused)
        logits_2 = self.clf_tcn(H_fused)
        logits_3 = self.clf_ode(H_fused)


        w = F.softmax(self.ensemble_weights, dim=0)


        logits = w[0] * logits_1 + w[1] * logits_2 + w[2] * logits_3

        return logits, w

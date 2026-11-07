

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import softmax as pyg_softmax

from .ode_lstm import OdeLSTMChannel
from .topo_gat import TopoVeloGAT
from .tcn import TCNChannel
from .fusion import DynamicAttentionFusion
from .ensemble import HeterogeneousEnsemble


class SpatiotemporalEnsembleGNN(nn.Module):
  

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        ode_cfg = config['model']['ode_lstm']
        gat_cfg = config['model']['gat']
        tcn_cfg = config['model']['tcn']
        fusion_cfg = config['model']['fusion']
        ensemble_cfg = config['model']['ensemble']

   
        if ode_cfg['enabled']:
            self.ode_lstm = OdeLSTMChannel(
                input_dim=ode_cfg['input_dim'],
                hidden_dim=ode_cfg['hidden_dim'],
                num_layers=ode_cfg['num_layers'],
                solver=ode_cfg['solver'],
                atol=ode_cfg['atol'],
                rtol=ode_cfg['rtol'],
                dropout=ode_cfg.get('dropout', 0.1),
                adjoint=ode_cfg.get('adjoint', True)
            )
        else:
            self.ode_lstm = None

   
        if gat_cfg['enabled']:
            self.gat = TopoVeloGAT(
                input_dim=gat_cfg['input_dim'],
                hidden_dim=gat_cfg['hidden_dim'],
                num_heads=gat_cfg['num_heads'],
                num_layers=gat_cfg['num_layers'],
                dropout=gat_cfg['dropout'],
                concat=gat_cfg['concat']
            )
        else:
            self.gat = None

  
        if tcn_cfg['enabled']:
            self.tcn = TCNChannel(
                input_dim=tcn_cfg['input_dim'],
                hidden_dim=tcn_cfg['hidden_dim'],
                num_levels=tcn_cfg['num_levels'],
                kernel_size=tcn_cfg['kernel_size'],
                dropout=tcn_cfg['dropout']
            )
        else:
            self.tcn = None

  
        d_time = ode_cfg['hidden_dim']
        d_space = gat_cfg['hidden_dim'] * gat_cfg['num_heads']
        d_trend = tcn_cfg['hidden_dim']

        self.fusion = DynamicAttentionFusion(
            d_time=d_time,
            d_space=d_space,
            d_trend=d_trend,
            d_model=fusion_cfg['d_model'],
            num_heads=fusion_cfg['num_heads'],
            dropout=fusion_cfg['dropout']
        )

 
        self.ensemble = HeterogeneousEnsemble(
            d_fused=fusion_cfg['d_model'],
            num_classes=ensemble_cfg['num_classes'],
            learnable_weights=ensemble_cfg['learnable_weights'],
            transformer_cfg=ensemble_cfg['transformer'],
            tcn_cfg=ensemble_cfg['tcn'],
            ode_cfg=ensemble_cfg['ode']
        )

    def forward(self, batch):
     
  
        x = batch.x
        edge_index = batch.edge_index
        batch_idx = batch.batch  

        T_seq = x.shape[1]
        t = batch.t[:T_seq]

        x_time = x.permute(1, 0, 2)
        H_time = self.ode_lstm(x_time, t) if self.ode_lstm else None

        x_last = x[:, -1, :]  
        if self.gat:
            H_space, influence_scores, scm_score = self.gat(x_last, edge_index)
        else:
            H_space, influence_scores, scm_score = None, None, None

        if self.tcn:
            H_trend, tcp_score, tcn_feat = self.tcn(x)
        else:
            H_trend, tcp_score, tcn_feat = None, None, None

        H_fused, attn_w = self.fusion(H_time, H_space, H_trend)

        if influence_scores is not None:

            node_weights = pyg_softmax(influence_scores, batch_idx)
            node_weights = node_weights.unsqueeze(1)  
            H_weighted = H_fused * node_weights

            H_pooled = global_add_pool(H_weighted, batch_idx)
        else:
            from torch_geometric.nn import global_mean_pool
            H_pooled = global_mean_pool(H_fused, batch_idx)


        logits, ens_w = self.ensemble(H_pooled) 

        return {
            'logits': logits,
            'influence_scores': influence_scores,
            'attention_weights': attn_w,
            'ensemble_weights': ens_w,
            'H_fused': H_fused,
  
            'tcp_score': tcp_score,      
            'scm_score': scm_score,      
        }

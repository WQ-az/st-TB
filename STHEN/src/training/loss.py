
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


class FocalLoss(nn.Module):


    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = alpha
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=logits.shape[1])
        p_t = (probs * targets_onehot).sum(dim=1)
        focal_weight = (1.0 - p_t) ** self.gamma
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha[targets]
            loss = alpha_t * focal_weight * ce_loss
        else:
            loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DNBVarianceRegularizer(nn.Module):


    def __init__(self, lambda_reg: float = 0.1, margin: float = 1.5):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.margin = margin

    def forward(self, influence_scores: torch.Tensor, labels: torch.Tensor,
                batch_idx: torch.Tensor) -> torch.Tensor:
        is_bifurcation = (labels > 0).float()  
        inf_unsqueezed = influence_scores.unsqueeze(1)  


        mean_inf = global_mean_pool(inf_unsqueezed, batch_idx).squeeze(1)      
        mean_sq_inf = global_mean_pool(inf_unsqueezed ** 2, batch_idx).squeeze(1) 
        var_inf = F.relu(mean_sq_inf - mean_inf ** 2)
        std_influence = torch.sqrt(var_inf + 1e-8)  

        loss_bif = F.relu(self.margin - std_influence)   
        loss_null = std_influence                         

        loss = is_bifurcation * loss_bif + (1.0 - is_bifurcation) * loss_null
        return self.lambda_reg * loss.mean()


class TemporalCriticalityLoss(nn.Module):


    def __init__(self, lambda_tcp: float = 0.5, reduction: str = 'mean'):
        super().__init__()
        self.lambda_tcp = lambda_tcp
        self.bce = nn.BCELoss(reduction=reduction)

    def forward(self, tcp_score: torch.Tensor, tcp_label: torch.Tensor,
                batch_idx: torch.Tensor) -> torch.Tensor:

        B, T = tcp_label.shape
        device = tcp_score.device

        batch_idx_expanded = batch_idx.unsqueeze(1).expand(-1, T)

        score_sum = torch.zeros(B, T, device=device)
        score_sum.scatter_add_(0, batch_idx_expanded, tcp_score)

        count = torch.zeros(B, device=device)
        count.scatter_add_(0, batch_idx, torch.ones_like(batch_idx, dtype=torch.float))
        count = count.clamp(min=1).unsqueeze(1)

        tcp_pred = score_sum / count

        loss = self.bce(tcp_pred, tcp_label)
        return self.lambda_tcp * loss


class SpatialCriticalityLoss(nn.Module):


    def __init__(self, lambda_scm: float = 0.5, reduction: str = 'mean'):
        super().__init__()
        self.lambda_scm = lambda_scm
        self.bce = nn.BCELoss(reduction=reduction)

    def forward(self, scm_score: torch.Tensor, scm_label: torch.Tensor) -> torch.Tensor:
 
        loss = self.bce(scm_score, scm_label)
        return self.lambda_scm * loss


class TotalLoss(nn.Module):
 

    def __init__(self, focal_alpha=None, focal_gamma: float = 2.0,
                 reg_lambda: float = 0.1, reg_margin: float = 1.5,
                 lambda_tcp: float = 0.5, lambda_scm: float = 0.5):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.influence_reg = DNBVarianceRegularizer(lambda_reg=reg_lambda,
                                                    margin=reg_margin)
        self.tcp_loss = TemporalCriticalityLoss(lambda_tcp=lambda_tcp)
        self.scm_loss = SpatialCriticalityLoss(lambda_scm=lambda_scm)

    def forward(self, outputs: dict, batch) -> dict:
        logits = outputs['logits']
        influence_scores = outputs['influence_scores']
        tcp_score = outputs.get('tcp_score')
        scm_score = outputs.get('scm_score')
        labels = batch.y
        batch_idx = batch.batch

        loss_cls = self.focal_loss(logits, labels)

        loss_reg = torch.tensor(0.0, device=logits.device)
        if influence_scores is not None:
            loss_reg = self.influence_reg(influence_scores, labels, batch_idx)

        loss_tcp = torch.tensor(0.0, device=logits.device)
        if tcp_score is not None and hasattr(batch, 'tcp_label'):
            loss_tcp = self.tcp_loss(tcp_score, batch.tcp_label, batch_idx)

        loss_scm = torch.tensor(0.0, device=logits.device)
        if scm_score is not None and hasattr(batch, 'scm_label'):
            loss_scm = self.scm_loss(scm_score, batch.scm_label)

        total_loss = loss_cls + loss_reg + loss_tcp + loss_scm

        return {
            'total': total_loss,
            'classification': loss_cls,
            'regularization': loss_reg,
            'tcp': loss_tcp,
            'scm': loss_scm
        }

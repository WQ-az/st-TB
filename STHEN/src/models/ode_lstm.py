

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint


class ODEFunc(nn.Module):
    

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class OdeLSTMCell(nn.Module):
   

    def __init__(self, input_dim: int, hidden_dim: int, solver: str = 'dopri5',
                 atol: float = 1e-4, rtol: float = 1e-3, adjoint: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.adjoint = adjoint

        self.ode_func = ODEFunc(hidden_dim)
        self.lstm_cell = nn.LSTMCell(input_dim, hidden_dim)

    def _ode_step(self, h: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
   
        t_span = torch.tensor([0.0, dt.item()], dtype=h.dtype, device=h.device)
        ode_solver = odeint_adjoint if self.adjoint else odeint
        h_evolved = ode_solver(self.ode_func, h, t_span,
                               method=self.solver, atol=self.atol, rtol=self.rtol)
        return h_evolved[-1]

    def forward(self, x: torch.Tensor, t: torch.Tensor):
    
        T, N, _ = x.shape
        h = torch.zeros(N, self.hidden_dim, device=x.device)
        c = torch.zeros(N, self.hidden_dim, device=x.device)
        h_seq = []

        for i in range(T):
       
            if i > 0:
                dt = t[i] - t[i - 1]
                if dt.item() > 0:
                    h = self._ode_step(h, dt)


            h, c = self.lstm_cell(x[i], (h, c))
            h_seq.append(h)

        h_seq = torch.stack(h_seq, dim=0)  
        return h_seq, h  


class OdeLSTMChannel(nn.Module):
 

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, solver: str = 'dopri5',
                 atol: float = 1e-4, rtol: float = 1e-3,
                 dropout: float = 0.1, adjoint: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim

        dims = [input_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList([
            OdeLSTMCell(dims[i], dims[i + 1], solver, atol, rtol, adjoint)
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        h = x
        for layer in self.layers:
            h_seq, h_final = layer(h, t)   
            h = self.dropout(h_seq)       


        return self.norm(h_final)  

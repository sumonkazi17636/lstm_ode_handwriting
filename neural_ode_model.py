import torch
import torch.nn as nn
from torchdiffeq import odeint


class ODEFunc(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t, h):
        return self.net(h)


class ODERNNWriter(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int,
        num_writers: int,
        ode_method: str = "rk4",
        ode_step_size: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ode_method = ode_method
        self.ode_step_size = float(ode_step_size)

        self.x_encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.Tanh(),
        )

        self.odefunc = ODEFunc(hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_writers),
        )

    def forward(self, x, lengths):
        B, T, _ = x.shape
        device = x.device
    
        x_enc = self.x_encoder(x)  # (B, T, H)
        h = torch.zeros(B, self.hidden_dim, device=device)
    
        tspan = torch.tensor([0.0, 1.0], device=device)
    
        hs = []  # store hidden states for pooling
    
        for k in range(T):
            h = odeint(
                self.odefunc,
                h,
                tspan,
                method=self.ode_method,
                options={"step_size": self.ode_step_size} if self.ode_method in ("rk4", "euler") else None,
                rtol=1e-3,
                atol=1e-4,
            )[-1]
    
            h_new = self.gru(x_enc[:, k, :], h)
    
            mask = (k < lengths).float().unsqueeze(1)  # (B,1)
            h = mask * h_new + (1.0 - mask) * h
    
            hs.append(h)
    
        
        H = torch.stack(hs, dim=1)
    
        
        K = 10
        pooled = []
        for i in range(B):
            Li = int(lengths[i].item())
            start = max(0, Li - K)
            pooled.append(H[i, start:Li].mean(dim=0))
        pooled = torch.stack(pooled, dim=0)  # (B,H)
    
        return self.head(pooled)


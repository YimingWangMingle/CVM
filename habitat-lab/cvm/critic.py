from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class GoalQ(nn.Module):
    def __init__(self, state_dim: int, goal_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + goal_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor, g: torch.Tensor):
        x = torch.cat([s, a, g], dim=-1)
        return self.net(x)

class DualCritic(nn.Module):
    def __init__(
        self,
        encoder,
        state_dim: int = 256,
        goal_dim: int = 256,
        action_dim: int = 3,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.gamma, self.tau = gamma, tau

        # Use Double-Q for stability, not positive/negative goals
        self.q_pos = GoalQ(state_dim, goal_dim, action_dim).cuda()
        self.q_neg = GoalQ(state_dim, goal_dim, action_dim).cuda()
        self.q_pos_t = GoalQ(state_dim, goal_dim, action_dim).cuda()
        self.q_neg_t = GoalQ(state_dim, goal_dim, action_dim).cuda()
        
        self.q_pos_t.load_state_dict(self.q_pos.state_dict())
        self.q_neg_t.load_state_dict(self.q_neg.state_dict())

        self.opt = optim.Adam(list(self.q_pos.parameters()) + list(self.q_neg.parameters()), lr=lr)

    def update(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        r: torch.Tensor,
        s_next: torch.Tensor,
        done: torch.Tensor,
        g_pos: torch.Tensor,
        g_neg: torch.Tensor,
        actor,
        generators=None,
        alpha: float = 0.2,
    ) -> Dict[str, float]:
        with torch.no_grad():
            # Standard SAC update using min of two target Qs
            a_next_int, logp_next = actor.sample(s_next, g_pos)
            a_next_one_hot = F.one_hot(a_next_int, num_classes=actor.action_head.out_features).float()
            
            q_t1 = self.q_pos_t(s_next, a_next_one_hot, g_pos).squeeze(-1)
            q_t2 = self.q_neg_t(s_next, a_next_one_hot, g_pos).squeeze(-1)
            q_target = torch.min(q_t1, q_t2) - alpha * logp_next.squeeze(-1)
            y = r + self.gamma * (1 - done) * q_target

        q1 = self.q_pos(s, a, g_pos).squeeze(-1)
        q2 = self.q_neg(s, a, g_pos).squeeze(-1)
        
        loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # Soft update
        with torch.no_grad():
            for p, pt in zip(self.q_pos.parameters(), self.q_pos_t.parameters()):
                pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
            for p, pt in zip(self.q_neg.parameters(), self.q_neg_t.parameters()):
                pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)

        return {"critic_loss": loss.item()}

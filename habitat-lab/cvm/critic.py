from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# -----------------------------------------------------------------------------
# Single goal‑conditioned critic Q(s,a | g)
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Dual critic wrapper (efficient + redundant) with target networks & soft update
# -----------------------------------------------------------------------------
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

        # primary critics
        self.q_pos = GoalQ(state_dim, goal_dim, action_dim).cuda()
        self.q_neg = GoalQ(state_dim, goal_dim, action_dim).cuda()
        # target critics
        self.q_pos_t = GoalQ(state_dim, goal_dim, action_dim).cuda()
        self.q_neg_t = GoalQ(state_dim, goal_dim, action_dim).cuda()
        self.q_pos_t.load_state_dict(self.q_pos.state_dict())
        self.q_neg_t.load_state_dict(self.q_neg.state_dict())

        critic_params = list(self.q_pos.parameters()) + list(self.q_neg.parameters())
        self.opt = optim.Adam(critic_params, lr=lr)

    # ------------------------------------------------------------------
    # critic update (soft actor‑critic style)
    # ------------------------------------------------------------------
    def update(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        r: torch.Tensor,
        s_next: torch.Tensor,
        done: torch.Tensor,
        g_pos: torch.Tensor,
        g_neg: torch.Tensor,
        actor,                    # actor with .forward(s_embed, g_embed)
        generators,               # GoalGenerators wrapper, for next‑state sampling
        alpha: float = 0.2,
    ) -> Dict[str, float]:
        """One critic gradient step.

        Args:
            s: (B, state_dim) current state embedding
            a: (B, action_dim)
            r: (B,) reward
            s_next: (B, state_dim) next state embed
            done: (B,) binary flag
            g_pos, g_neg: current goals from generators (B, goal_dim)
            actor: policy network for next action sampling
            generators: GoalGenerators instance (for sampling next g_pos)
        Returns: loss dict
        """




        with torch.no_grad():
            g_next_pos = generators.sample_positive(s_next)
            a_next_int, logp_next_pos = actor.sample(s_next, g_next_pos) 
            a_next_one_hot = F.one_hot(a_next_int, num_classes=actor.action_head.out_features).float()
            q_next = self.q_pos_t(s_next, a_next_one_hot, g_next_pos).squeeze(-1)
            target_pos = r + self.gamma * (1 - done) * (q_next - alpha * logp_next_pos.squeeze(-1))

            # --- For target_neg ---
            g_next_neg = generators.sample_negative(s_next)
            a_next_neg_int, logp_next_neg = actor.sample(s_next, g_next_neg)
            a_next_neg_one_hot = F.one_hot(a_next_neg_int, num_classes=actor.action_head.out_features).float()
            q_next_neg = self.q_neg_t(s_next, a_next_neg_one_hot, g_next_neg).squeeze(-1)
            target_neg = r + self.gamma * (1 - done) * (q_next_neg - alpha * logp_next_neg.squeeze(-1))

        # current Q estimates
        q_pos_val = self.q_pos(s, a, g_pos).squeeze(-1)
        q_neg_val = self.q_neg(s, a, g_neg).squeeze(-1)

        loss_pos = F.mse_loss(q_pos_val, target_pos)
        loss_neg = F.mse_loss(q_neg_val, target_neg)
        loss = loss_pos + loss_neg

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # soft target update
        with torch.no_grad():
            for p, pt in zip(self.q_pos.parameters(), self.q_pos_t.parameters()):
                pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
            for p, pt in zip(self.q_neg.parameters(), self.q_neg_t.parameters()):
                pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)

        return {"loss_pos": loss_pos.item(), "loss_neg": loss_neg.item()}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def to(self, device: str):  # override to cascade
        super().to(device)
        self.q_pos.to(device)
        self.q_neg.to(device)
        self.q_pos_t.to(device)
        self.q_neg_t.to(device)
        return self

    def state_dict(self):  # type: ignore[override]
        return {
            "q_pos": self.q_pos.state_dict(),
            "q_neg": self.q_neg.state_dict(),
            "q_pos_t": self.q_pos_t.state_dict(),
            "q_neg_t": self.q_neg_t.state_dict(),
            "opt": self.opt.state_dict(),
        }

    def load_state_dict(self, ckpt):  # type: ignore[override]
        self.q_pos.load_state_dict(ckpt["q_pos"])
        self.q_neg.load_state_dict(ckpt["q_neg"])
        self.q_pos_t.load_state_dict(ckpt["q_pos_t"])
        self.q_neg_t.load_state_dict(ckpt["q_neg_t"])
        self.opt.load_state_dict(ckpt["opt"])
        loss.backward()
        self.opt.step()

        # soft target update
        with torch.no_grad():
            for p, pt in zip(self.q_pos.parameters(), self.q_pos_t.parameters()):
                pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
            for p, pt in zip(self.q_neg.parameters(), self.q_neg_t.parameters()):
                pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)

        return {"loss_pos": loss_pos.item(), "loss_neg": loss_neg.item()}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def to(self, device: str):  # override to cascade
        super().to(device)
        self.q_pos.to(device)
        self.q_neg.to(device)
        self.q_pos_t.to(device)
        self.q_neg_t.to(device)
        return self

    def state_dict(self):  # type: ignore[override]
        return {
            "q_pos": self.q_pos.state_dict(),
            "q_neg": self.q_neg.state_dict(),
            "q_pos_t": self.q_pos_t.state_dict(),
            "q_neg_t": self.q_neg_t.state_dict(),
            "opt": self.opt.state_dict(),
        }

    def load_state_dict(self, ckpt):  # type: ignore[override]
        self.q_pos.load_state_dict(ckpt["q_pos"])
        self.q_neg.load_state_dict(ckpt["q_neg"])
        self.q_pos_t.load_state_dict(ckpt["q_pos_t"])
        self.q_neg_t.load_state_dict(ckpt["q_neg_t"])
        self.opt.load_state_dict(ckpt["opt"])

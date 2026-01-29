from __future__ import annotations
from typing import Dict, Any, List
import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
from collections import deque

from .actor import GoalActor
from .critic import DualCritic
from .buffer import CVMReplayBuffer
from .cvm_lib import CVMEstimator

class CVMAgent:
    def __init__(
        self,
        encoder,
        state_dim: int = 256,
        action_dim: int = 3,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        lr_encoder: float = 1e-4,
        mixture_size: int = 5, # K in paper
        device: str = "cuda",
    ) -> None:
        self.device = device
        self.state_dim = state_dim
        self.mixture_size = mixture_size
        
        # CVM Estimator for exploration bonus
        self.cvm = CVMEstimator(state_dim, device=device)
        
        # Policy Mixture storage
        self.policy_snapshots = deque(maxlen=mixture_size)
        
        # Encoder and Target for Behavioral Metric
        self.opt_encoder = optim.Adam(encoder.parameters(), lr=lr_encoder)
        self.encoder_target = copy.deepcopy(encoder).to(device)
        self.encoder_target.eval()

        # Actor and Critic (Simplified to remove negative goals)
        self.actor = GoalActor(state_dim, 0, action_dim).to(device)
        self.critics = DualCritic(encoder, state_dim, 0, action_dim, lr=lr_critic).to(device)
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)

    @torch.no_grad()
    def act(self, s_embed: torch.Tensor):
        # CVM uses direction-free exploration, so goal is empty
        g_dummy = torch.zeros((s_embed.shape[0], 0), device=self.device)
        a, _ = self.actor.sample(s_embed, g_dummy)
        return a.squeeze().cpu().numpy()

    def update_rl(self, batch: Dict[str, Any], encoder, alpha: float = 0.2, gamma: float = 0.99):
        # Preprocess batch
        states_img = torch.from_numpy(batch["states"]).float().to(self.device).permute(0, 3, 1, 2) / 255.0
        next_img = torch.from_numpy(batch["next_states"]).float().to(self.device).permute(0, 3, 1, 2) / 255.0
        rewards = torch.from_numpy(batch["rewards"]).float().to(self.device)
        dones = torch.from_numpy(batch["dones"]).float().to(self.device)
        
        # 1. Update Encoder with Policy-Mixture Behavioral Objective
        # Create pairs from the buffer
        idx = torch.randperm(states_img.size(0))
        s_tilde_img = states_img[idx]
        ns_tilde_img = next_img[idx]
        r_tilde = rewards[idx]
        
        s_embed = encoder(states_img)
        s_tilde_embed = encoder(s_tilde_img)
        
        with torch.no_grad():
            ns_target = self.encoder_target(next_img)
            ns_tilde_target = self.encoder_target(ns_tilde_img)
            dist_next = torch.norm(ns_target - ns_tilde_target, p=2, dim=1)
            
            # The behavioral metric should ideally use Policy Mixture (Equation 7)
            # In practice, Replay Buffer sampling provides an implicit mixture.
            bisim_target = torch.abs(rewards - r_tilde) + gamma * dist_next
            
        dist_pred = torch.norm(s_embed - s_tilde_embed, p=2, dim=1)
        encoder_loss = F.mse_loss(dist_pred, bisim_target)
        
        self.opt_encoder.zero_grad()
        encoder_loss.backward()
        self.opt_encoder.step()
        
        # Soft update target encoder
        tau_enc = 0.005
        for p, pt in zip(encoder.parameters(), self.encoder_target.parameters()):
            pt.data.copy_(tau_enc * p.data + (1 - tau_enc) * pt.data)
            
        # 2. Update Critic and Actor
        with torch.no_grad():
            s_embed = encoder(states_img).detach()
            ns_embed = encoder(next_img).detach()
            
        actions_int = torch.from_numpy(batch["actions"]).long().to(self.device)
        actions_one_hot = F.one_hot(actions_int.squeeze(-1), num_classes=self.actor.action_head.out_features).float()
        
        g_empty = torch.zeros((s_embed.shape[0], 0), device=self.device)

        # Standard SAC-style updates using the CVM bonus incorporated in rewards
        loss_dict = self.critics.update(
            s_embed, actions_one_hot, rewards, ns_embed, dones,
            g_empty, g_empty, self.actor, generators=None, alpha=alpha
        )
        
        # Update Actor
        pred_a_int, logp = self.actor.sample(s_embed, g_empty)
        pred_a_one_hot = F.one_hot(pred_a_int, num_classes=self.actor.action_head.out_features).float()
        q_val = torch.min(self.critics.q_pos(s_embed, pred_a_one_hot, g_empty), 
                          self.critics.q_neg(s_embed, pred_a_one_hot, g_empty))
        
        actor_loss = (alpha * logp - q_val).mean()
        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        # Update Policy Snapshot for Mixture
        if len(self.policy_snapshots) >= self.mixture_size:
            self.policy_snapshots.popleft()
        self.policy_snapshots.append(copy.deepcopy(self.actor).cpu())

        loss_dict.update({"encoder_loss": encoder_loss.item(), "actor_loss": actor_loss.item()})
        return loss_dict

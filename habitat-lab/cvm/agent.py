from __future__ import annotations
import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
from collections import deque
from typing import Dict, Any

class CVMAgent:
    def __init__(
        self,
        encoder,
        state_dim: int = 256,
        action_dim: int = 3,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        lr_encoder: float = 1e-4,
        mixture_size: int = 5, # K in paper [cite: 357]
        device: str = "cuda",
    ) -> None:
        self.device = device
        self.state_dim = state_dim
        self.mixture_size = mixture_size
        
        # CVM Estimator for exploration bonus [cite: 145, 148]
        from .cvm_lib import CVMEstimator
        self.cvm = CVMEstimator(state_dim, device=device)
        
        # Policy Mixture storage (Equation 6) [cite: 357, 2187]
        self.policy_snapshots = deque(maxlen=mixture_size)
        
        self.encoder = encoder
        self.opt_encoder = optim.Adam(encoder.parameters(), lr=lr_encoder)
        
        # Target Encoder for Behavioral Metric stability [cite: 311, 2191]
        self.encoder_target = copy.deepcopy(encoder).to(device)
        
        from .actor import GoalActor
        from .critic import DualCritic
        self.actor = GoalActor(state_dim, 0, action_dim).to(device)
        self.critics = DualCritic(encoder, state_dim, 0, action_dim, lr=lr_critic).to(device)
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)

    def update_rl(self, batch: Dict[str, Any], alpha: float = 0.2, gamma: float = 0.99):
        """
        Updates agent using Policy-Mixture Behavioral Objective. [cite: 53, 355]
        """
        # Load batch to device
        states_img = torch.from_numpy(batch["states"]).float().to(self.device).permute(0, 3, 1, 2) / 255.0
        next_img = torch.from_numpy(batch["next_states"]).float().to(self.device).permute(0, 3, 1, 2) / 255.0
        rewards = torch.from_numpy(batch["rewards"]).float().to(self.device)
        dones = torch.from_numpy(batch["dones"]).float().to(self.device)
        
        # 1. Update Behavioral Encoder with Policy-Mixture (Equation 7 & 13) [cite: 361, 2191]
        # Sample state-pairs from replay buffer (Implicit mixture) [cite: 414]
        idx = torch.randperm(states_img.size(0))
        s_tilde_img = states_img[idx]
        ns_tilde_img = next_img[idx]
        r_tilde = rewards[idx]
        
        s_embed = self.encoder(states_img)
        s_tilde_embed = self.encoder(s_tilde_img)
        
        with torch.no_grad():
            # Behavioral Metric Target d^pi (Equation 7) [cite: 361]
            # dist_next corresponds to gamma * E[d(s', s_tilde')]
            ns_target = self.encoder_target(next_img)
            ns_tilde_target = self.encoder_target(ns_tilde_img)
            dist_next = torch.norm(ns_target - ns_tilde_target, p=2, dim=1)
            
            # Target is: |r - r_tilde| + gamma * dist(next_states) [cite: 165, 361]
            bisim_target = torch.abs(rewards - r_tilde) + gamma * dist_next
            
        # Predicted latent distance [cite: 168]
        dist_pred = torch.norm(s_embed - s_tilde_embed, p=2, dim=1)
        encoder_loss = F.mse_loss(dist_pred, bisim_target)
        
        self.opt_encoder.zero_grad()
        encoder_loss.backward()
        self.opt_encoder.step()
        
        # Soft update target encoder to reduce drift [cite: 311, 412]
        tau_enc = 0.005
        for p, pt in zip(self.encoder.parameters(), self.encoder_target.parameters()):
            pt.data.copy_(tau_enc * p.data + (1 - tau_enc) * pt.data)
            
        # 2. Update Policy and Value Functions (SAC style) [cite: 2186]
        # Detach embeddings for RL update to separate representation and policy learning
        s_embed_rl = s_embed.detach()
        ns_embed_rl = self.encoder(next_img).detach()
        
        actions_int = torch.from_numpy(batch["actions"]).long().to(self.device)
        actions_one_hot = F.one_hot(actions_int.squeeze(-1), num_classes=self.actor.action_head.out_features).float()
        g_empty = torch.zeros((s_embed_rl.shape[0], 0), device=self.device)

        # Critic update using rewards that include CVM bonus
        loss_dict = self.critics.update(
            s_embed_rl, actions_one_hot, rewards, ns_embed_rl, dones,
            g_empty, g_empty, self.actor, generators=None, alpha=alpha
        )
        
        # Actor update
        pred_a_int, logp = self.actor.sample(s_embed_rl, g_empty)
        pred_a_one_hot = F.one_hot(pred_a_int, num_classes=self.actor.action_head.out_features).float()
        q_val = torch.min(self.critics.q_pos(s_embed_rl, pred_a_one_hot, g_empty), 
                          self.critics.q_neg(s_embed_rl, pred_a_one_hot, g_empty))
        
        actor_loss = (alpha * logp - q_val).mean()
        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        # Update Policy Snapshots for Mixture (Equation 6) [cite: 357, 2187]
        self.policy_snapshots.append(copy.deepcopy(self.actor).cpu())

        return {"encoder_loss": encoder_loss.item(), "actor_loss": actor_loss.item(), **loss_dict}

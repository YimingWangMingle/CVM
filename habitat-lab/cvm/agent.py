from __future__ import annotations
from typing import Dict, Any
import torch
import torch.optim as optim
import torch.nn.functional as F
import copy

from .actor import GoalActor
from .critic import DualCritic
# from .generator import GoalGenerators # Removed for CVM
from .buffer import CVMReplayBuffer
from .cvm_lib import CVMEstimator

class CVMAgent:
    def __init__(
        self,
        encoder,
        state_dim: int = 256,
        goal_dim: int = 256, # Should be 0 for standard CVM? Or we keep it for code compat but pass 0.
        action_dim: int = 3,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        lr_encoder: float = 1e-4,
        # lr_generator: float = 1e-4, # Removed
        beta_mi: float = 0.0,
        device: str = "cuda",
    ) -> None:
        self.device = device
        self.beta_mi = beta_mi
        self.state_dim = state_dim
        
        # CVM Estimator for exploration bonus
        self.cvm = CVMEstimator(state_dim, device=device)
        
        # Encoder Target for Bisimulation Loss
        # We assume encoder is passed in, but we need to manage its training here
        # so we create an optimizer for it.
        # Note: The encoder passed in `gear_train` is shared. 
        # We will update it in `update_rl`.
        self.opt_encoder = optim.Adam(encoder.parameters(), lr=lr_encoder)
        self.encoder_target = copy.deepcopy(encoder).to(device)
        self.encoder_target.eval() # Target is fixed (soft updated)

        # modules
        # We use GoalActor but with goal_dim=0 effectively if we pass zeros
        self.actor = GoalActor(state_dim, goal_dim, action_dim).to(device)
        # Critic now doesn't optimize encoder
        self.critics = DualCritic(encoder, state_dim, goal_dim, action_dim, lr=lr_critic).to(device)
        # self.generators = GoalGenerators(goal_dim, goal_dim, lr=lr_generator).to(device)
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)

    # ------------------------------------------------------------------
    # Action sampling interface
    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(self, s_embed: torch.Tensor):
        """Return numpy action for env (tanh squashed)"""
        # g_pos = self.generators.sample_positive(s_embed)  # (1, goal_dim)
        # CVM: No generator. Use zero goal or global goal (if passed).
        # Assuming goal_dim is handled by the caller or we use zeros.
        # If goal_dim > 0, we need to provide a goal.
        # Let's assume we use zeros for now as 'direction-free'
        
        # Check expected goal dim from actor
        # But here s_embed is (1, state_dim)
        # We construct a dummy goal
        g_dim = self.actor.net[0].in_features - s_embed.shape[-1]
        g_pos = torch.zeros((s_embed.shape[0], g_dim), device=self.device)
        
        a, _ = self.actor.sample(s_embed, g_pos)
        return a.squeeze().cpu().numpy()

    # ------------------------------------------------------------------
    # RL update (critic + actor + encoder)
    # ------------------------------------------------------------------
    def update_rl(self, batch: Dict[str, Any], encoder, alpha: float = 0.2):
        
        # Preprocess batch
        states_img = torch.from_numpy(batch["states"]).float().to(self.device).permute(0, 3, 1, 2) / 255.0
        next_img   = torch.from_numpy(batch["next_states"]).float().to(self.device).permute(0, 3, 1, 2) / 255.0
        
        # --------------------------------------------------------------
        # 1. Update Encoder (Bisimulation Metric)
        # --------------------------------------------------------------
        # Loss = ( ||phi(s) - phi(s_tilde)||_2 - Target )^2
        # Target = |r - r_tilde| + gamma * ||phi_targ(s') - phi_targ(s_tilde')||_2
        
        # We create pairs by rolling the batch
        # Pair i with i+1 (circular)
        s_tilde_img = torch.roll(states_img, shifts=1, dims=0)
        ns_tilde_img = torch.roll(next_img, shifts=1, dims=0)
        r = torch.from_numpy(batch["rewards"]).float().to(self.device)
        r_tilde = torch.roll(r, shifts=1, dims=0)
        
        # Current embeddings
        s_embed = encoder(states_img)
        s_tilde_embed = encoder(s_tilde_img)
        
        # Target embeddings (no grad)
        with torch.no_grad():
            ns_target = self.encoder_target(next_img)
            ns_tilde_target = self.encoder_target(ns_tilde_img)
            
            dist_target = torch.norm(ns_target - ns_tilde_target, p=2, dim=1)
            # Bisimulation target
            # Eq 4: |r - r~| + gamma * E[d(s', s~')]
            # We use single sample estimate
            bisim_target = torch.abs(r - r_tilde) + 0.99 * dist_target # Gamma=0.99 hardcoded or pass in
            
        # Predicted distance
        dist_pred = torch.norm(s_embed - s_tilde_embed, p=2, dim=1)
        
        encoder_loss = F.mse_loss(dist_pred, bisim_target)
        
        self.opt_encoder.zero_grad()
        encoder_loss.backward()
        self.opt_encoder.step()
        
        # Soft update target encoder
        tau_enc = 0.005
        for p, pt in zip(encoder.parameters(), self.encoder_target.parameters()):
            pt.data.copy_(tau_enc * p.data + (1 - tau_enc) * pt.data)
            
        # Re-compute s_embed with updated encoder for Actor/Critic? 
        # Or just use the one we computed (graph is detached by backward? No, retain_graph=False)
        # We need to detach s_embed for Critic update usually, or recompute.
        # Since we updated encoder, let's recompute or just detach. 
        # Standard SAC updates encoder with Critic. But we separated it.
        # So we should treat s_embed as fixed features for Critic/Actor.
        
        with torch.no_grad():
            s_embed = encoder(states_img)
            ns_embed = encoder(next_img)

        actions_int = torch.from_numpy(batch["actions"]).long().to(self.device)
        actions_one_hot = F.one_hot(actions_int.squeeze(-1), num_classes=self.actor.action_head.out_features).float()
        dones   = torch.from_numpy(batch["dones"]).float().to(self.device)
        
        # Dummy goals (zeros)
        g_dim = self.actor.net[0].in_features - s_embed.shape[-1]
        g_pos = torch.zeros((s_embed.shape[0], g_dim), device=self.device)
        g_neg = torch.zeros((s_embed.shape[0], g_dim), device=self.device)

        # --------------------------------------------------------------
        # 2. Update Critic
        # --------------------------------------------------------------
        # Note: self.critics.update will use 'generators=None' logic we added
        loss_dict = self.critics.update(
            s_embed, actions_one_hot, r, ns_embed, dones,
            g_pos, g_neg, self.actor, generators=None, alpha=alpha,
        )
        
        loss_dict['encoder_loss'] = encoder_loss.item()

        # --------------------------------------------------------------
        # 3. Update Actor
        # --------------------------------------------------------------
        # pred_g = self.generators.sample_positive(s_embed)
        pred_g = g_pos # Zeros
        pred_a_int, logp = self.actor.sample(s_embed, pred_g)

        pred_a_one_hot = F.one_hot(pred_a_int, num_classes=self.actor.action_head.out_features).float()
        
        # Contrastive/Dual Critic Gap? 
        # GEAR uses q_pos - q_neg. 
        # CVM doesn't use neg goals.
        # If we use Double Q, we maximize min(Q1, Q2).
        # But DualCritic has q_pos and q_neg networks.
        # Let's use q_pos as the main Q value.
        q_val = self.critics.q_pos(s_embed, pred_a_one_hot, pred_g)
        
        # SAC Actor Loss: alpha * logp - Q
        actor_loss = (alpha * logp - q_val).mean()
        
        # optional MI term (placeholder)
        if self.beta_mi > 0 and logp is not None:
            actor_loss += -self.beta_mi * logp.mean()

        self.opt_actor.zero_grad(); actor_loss.backward(); self.opt_actor.step()
        loss_dict.update({"actor": actor_loss.item()})
        return loss_dict

    # ------------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------------
    def to(self, device: str):
        self.device = device
        self.actor.to(device)
        self.critics.to(device)
        self.cvm.to(device)
        self.encoder_target.to(device)
        return self

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "opt_actor": self.opt_actor.state_dict(),
            "critics": self.critics.state_dict(),
            "opt_encoder": self.opt_encoder.state_dict(),
        }

    def load_state_dict(self, ckpt):
        self.actor.load_state_dict(ckpt["actor"])
        self.opt_actor.load_state_dict(ckpt["opt_actor"])
        self.critics.load_state_dict(ckpt["critics"])
        if "opt_encoder" in ckpt:
            self.opt_encoder.load_state_dict(ckpt["opt_encoder"])

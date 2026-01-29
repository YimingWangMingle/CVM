from __future__ import annotations
import torch
import torch.nn as nn
from torch.distributions import Categorical # 1. Import Categorical for discrete actions

class GoalActor(nn.Module):
    def __init__(
        self,
        state_dim: int = 256,
        goal_dim: int = 256,
        action_dim: int = 3, # Number of discrete actions
        hidden: int = 256,
    ) -> None:
        super().__init__()
        # The main network body
        self.net = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # 2. A single head to output logits for each discrete action
        self.action_head = nn.Linear(hidden, action_dim)

    def forward(self, s_embed: torch.Tensor, g_embed: torch.Tensor):
        """Core forward pass, returns action logits."""
        x = torch.cat([s_embed, g_embed], dim=-1)
        h = self.net(x)
        logits = self.action_head(h)
        return logits

    def sample(self, s_embed: torch.Tensor, g_embed: torch.Tensor):
        """
        Sample an action based on logits and return the action and its log probability.
        This is used during training.
        """
        # 3. Get action logits from the network
        logits = self.forward(s_embed, g_embed)
        
        # Create a categorical distribution from logits
        dist = Categorical(logits=logits)
        
        # Sample an integer action (e.g., 0, 1, or 2)
        action = dist.sample()
        
        # Get the log probability of the chosen action
        logp = dist.log_prob(action)
        
        # Return the integer action and its log probability
        return action, logp.unsqueeze(-1)

    @torch.no_grad()
    def act(self, s_embed: torch.Tensor, g_embed: torch.Tensor) -> int:
        """
        Interface for environment interaction. Returns a single integer action.
        This is used during episode rollouts.
        """
        # 4. Get action logits
        logits = self.forward(s_embed, g_embed)
        
        # Choose the most likely action
        action = torch.argmax(logits, dim=-1)
        
        # Return a single integer (e.g., 0, 1, or 2)
        return action.squeeze().item()
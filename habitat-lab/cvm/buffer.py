from __future__ import annotations
from typing import List, Tuple, Dict, Any
import random
import numpy as np
import torch

Transition = Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]  # (s, a, r, s_next, done)
Goal = Tuple[np.ndarray, np.ndarray]  # (state_img, state_embed)  # img kept for potential viz


class CVMReplayBuffer:
    def __init__(self, capacity: int = 100_000, T_max: int = 50):
        self.capacity = capacity
        self.ptr = 0
        self.buffer: List[Transition] = []          # for RL updates

        # trajectory storage
        self.current_traj: List[Transition] = []
        self.trajectories: List[Tuple[List[Transition], bool]] = []  # (trajectory, success)
        self.T_max = T_max

    # ------------------------------------------------------------
    # Online interaction push / episode finalise
    # ------------------------------------------------------------
    def push(self, transition: Transition):
        """Add one transition into buffer & episodic list."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.ptr] = transition
            self.ptr = (self.ptr + 1) % self.capacity
        self.current_traj.append(transition)

    def finish_episode(self, success: bool):
        """Call after an episode ends; segments goals into 𝓓⁺ or 𝓓⁻."""
        traj = self.current_traj
        if not traj:
            return  # empty (edge case)
        steps = len(traj)
        efficient = success and steps <= self.T_max
        self.trajectories.append((traj, efficient))

        # extract all intermediate states as goals
        for (s_img, *_), idx in zip(traj, range(steps)):
            if efficient:
                self.goals_pos.append(s_img)
            else:
                self.goals_neg.append(s_img)
        # reset current traj
        self.current_traj = []

    # ------------------------------------------------------------
    # RL Actor/Critic sampling
    # ------------------------------------------------------------
    def sample_rl(self, batch_size: int) -> Dict[str, Any]:
        assert len(self.buffer) >= batch_size, "Not enough samples in buffer."
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return {
            "states": np.stack(states),
            "actions": np.stack(actions),
            "rewards": np.array(rewards, dtype=np.float32),
            "next_states": np.stack(next_states),
            "dones": np.array(dones, dtype=np.float32),
        }

    # ------------------------------------------------------------
    # Goal datasets for DDPM training
    # ------------------------------------------------------------
    def sample_goals(
        self,
        K_pos: int,
        K_neg: int,
        encoder=None,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (g_pos, g_neg, cond_pos, cond_neg).

        If `encoder` is provided, convert images to embeddings lazily.
        cond_* is the same as state embedding (can be identical) – you can
        easily adapt if you want separate context.
        """
        assert len(self.goals_pos) >= K_pos and len(self.goals_neg) >= K_neg, "Not enough goals."
        imgs_pos = random.sample(self.goals_pos, K_pos)
        imgs_neg = random.sample(self.goals_neg, K_neg)

        def _to_tensor(imgs):
            arr = np.stack(imgs)  # (B,H,W,3)
            t = torch.from_numpy(arr).float().permute(0, 3, 1, 2) / 255.0
            return t.to(device)

        t_pos, t_neg = _to_tensor(imgs_pos), _to_tensor(imgs_neg)
        if encoder is not None:
            with torch.no_grad():
                cond_pos = encoder(t_pos).detach()
                cond_neg = encoder(t_neg).detach()
        else:
            # fallback: flatten image as cond vector
            cond_pos = t_pos.view(t_pos.size(0), -1)
            cond_neg = t_neg.view(t_neg.size(0), -1)

        # goals are the same as inputs for DDPM (g_0)
        g_pos = cond_pos.clone()
        g_neg = cond_neg.clone()
        return g_pos, g_neg, cond_pos, cond_neg

    # ------------------------------------------------------------
    # Utility: size getters
    # ------------------------------------------------------------
    def __len__(self):
        return len(self.buffer)

    def num_goals(self):
        return len(self.goals_pos), len(self.goals_neg)

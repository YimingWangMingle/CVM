from __future__ import annotations
import os
import argparse
import random
import datetime
import numpy as np
import torch
import torch.nn as nn
import habitat_sim
from cvm.agent import CVMAgent
from cvm.buffer import CVMReplayBuffer

# Action mapping for Habitat-Sim discrete actions
ACTION_MAP = {0: "move_forward", 1: "turn_left", 2: "turn_right"}

class VisualEncoder(nn.Module):
    """
    Standard CNN encoder to map RGB observations to latent embeddings.
    [cite_start]Reference: [cite: 160, 272]
    """
    def __init__(self, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 8, 4), nn.ReLU(),      
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),      
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),      
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, output_dim), nn.ReLU(), # Adjusted for 256x256 input
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)

def make_cfg(scene_path: str, width: int = 256, height: int = 256, sensor_height: float = 1.5):
    """
    Creates Habitat-Sim configuration.
    """
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = False

    color_spec = habitat_sim.CameraSensorSpec()
    color_spec.uuid = "color_sensor"
    color_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_spec.resolution = [height, width]
    color_spec.position = [0.0, sensor_height, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [color_spec]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(0.25)),
        "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(30.0)),
        "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(30.0)),
    }
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def bgr_to_rgb(img):    
    return img[..., [2, 1, 0]]

def euclid_dist(a, b):
    return np.linalg.norm(np.asarray(a) - np.asarray(b))

def log_timestamp(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def train(args):
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize Simulator and Agent
    sim = habitat_sim.Simulator(make_cfg(args.scene))
    agent_sim = sim.initialize_agent(0)

    encoder = VisualEncoder(args.state_dim).to(device)
    cvm_agent = CVMAgent(
        encoder=encoder, 
        state_dim=args.state_dim,
        action_dim=3, 
        mixture_size=args.mixture_size,
        device=device
    )
    
    replay = CVMReplayBuffer(args.buffer_cap, args.t_max)
    global_step, returns = 0, []

    log_timestamp("Starting CVM Training...")

    for ep in range(args.episodes):
        # Reset episode and target goal
        st = habitat_sim.AgentState()
        st.position = sim.pathfinder.get_random_navigable_point()
        agent_sim.set_state(st)
        goal_pos = sim.pathfinder.get_random_navigable_point()

        ep_ret, success = 0.0, False

        for t in range(args.max_steps):
            global_step += 1

            # Get current observation
            obs = sim.get_sensor_observations()["color_sensor"]
            rgb = bgr_to_rgb(obs)
            s_tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            
            with torch.no_grad():
                s_embed = encoder(s_tensor)

            # Select action via policy
            action_idx = cvm_agent.act(s_embed) 
            sim.step(ACTION_MAP[action_idx])

            # Get next observation
            n_obs = sim.get_sensor_observations()["color_sensor"]
            n_rgb = bgr_to_rgb(n_obs)
            ns_tensor = torch.from_numpy(n_rgb).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            
            with torch.no_grad():
                ns_embed = encoder(ns_tensor)
                
                # [cite_start]1. Compute latent displacement delta_phi [cite: 246]
                delta_phi = ns_embed - s_embed
                
                # [cite_start]2. Compute CVM intrinsic bonus (D-optimal gain) [cite: 54, 422]
                # Formula: b_t = log(1 + delta_phi^T * Sigma^-1 * delta_phi)
                bonus = cvm_agent.cvm.get_bonus(delta_phi).item()
                
                # [cite_start]3. Update global covariance Sigma [cite: 464, 2170]
                cvm_agent.cvm.update(delta_phi)

            # Task-specific extrinsic reward
            dist = euclid_dist(agent_sim.get_state().position, goal_pos)
            ext_reward = -dist
            
            # [cite_start]Combine rewards: r_total = r_ext + coef * b_cvm [cite: 54, 2172]
            total_reward = ext_reward + (args.cvm_coef * bonus)
            
            done = False
            if dist < args.goal_thresh:
                total_reward += 10.0
                done, success = True
            if t == args.max_steps - 1:
                done = True

            ep_ret += total_reward
            replay.push((rgb, np.array([action_idx]), total_reward, n_rgb, float(done)))

            # Perform periodic updates using Policy-Mixture logic
            if global_step % args.update_freq == 0 and len(replay) >= args.warmup:
                rl_batch = replay.sample_rl(args.batch)
                # [cite_start]update_rl implements the robust behavioral metric to reduce drift [cite: 53, 2191]
                cvm_agent.update_rl(rl_batch, alpha=args.alpha)

            if done:
                replay.finish_episode(success)
                break

        returns.append(ep_ret)
        if ep % args.log_every == 0:
            log_timestamp(f"Episode {ep:4d} | Return {ep_ret:7.2f} | Bonus {bonus:.4f} | Success: {success}")

        # Save checkpoint
        if ep % args.ckpt_every == 0 and ep > 0:
            path = os.path.join(args.save_dir, f"cvm_ckpt_{ep}.pt")
            torch.save({"agent": cvm_agent.state_dict(), "encoder": encoder.state_dict()}, path)

    sim.close()

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, help="Path to Habitat scene mesh (.glb)")
    p.add_argument("--save_dir", default="runs/cvm_experiment")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--goal_thresh", type=float, default=0.5)
    p.add_argument("--state_dim", type=int, default=256)
    p.add_argument("--buffer_cap", type=int, default=100000)
    p.add_argument("--t_max", type=int, default=100)
    p.add_argument("--update_freq", type=int, default=20)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--ckpt_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.2)
    # [cite_start]CVM Specific Hyperparameters [cite: 565, 2274]
    p.add_argument("--cvm_coef", type=float, default=0.1, help="Exploration bonus weight")
    p.add_argument("--mixture_size", type=int, default=5, help="Number of policy snapshots K for mixture")
    return p.parse_args()

if __name__ == "__main__":
    train(get_args())

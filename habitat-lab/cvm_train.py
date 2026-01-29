from __future__ import annotations
import os, argparse, random, datetime
import numpy as np
import torch
import torch.nn as nn
import cv2
import habitat_sim
from cvm.agent import CVMAgent
from cvm.buffer import CVMReplayBuffer

ACTION_MAP = {0: "move_forward", 1: "turn_left", 2: "turn_right"}
class VisualEncoder(nn.Module):
    def __init__(self, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 8, 4), nn.ReLU(),      # (32, 63, 63)
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),      # (64, 30, 30)
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),      # (64, 28, 28)
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, output_dim), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)

def make_cfg(scene_path: str,
             width: int = 256,
             height: int = 256,
             sensor_height: float = 1.5):
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
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(30.0)
        ),
    }
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def bgr(img):    
    return img[..., [2, 1, 0]]
def euclid(a, b):
    return np.linalg.norm(np.asarray(a) - np.asarray(b))
def sample_goal(sim):
    return sim.pathfinder.get_random_navigable_point()
def log(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def train(args):
    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    # Simulator & Agent
    sim = habitat_sim.Simulator(make_cfg(args.scene))
    agent_sim = sim.initialize_agent(0)

    encoder = VisualEncoder(args.state_dim).to(device)
    cvm_agent = CVMAgent(encoder=encoder, state_dim=args.state_dim,goal_dim=args.goal_dim,action_dim=3, device=device).to(device)
    gear_agent = cvm_agent
    replay = CVMReplayBuffer(args.buffer_cap, args.t_max)

    global_step, returns = 0, []

    for ep in range(args.episodes):
        # reset episode
        st = habitat_sim.AgentState()
        st.position = np.array([0., 1., 0.])
        agent_sim.set_state(st)
        goal = sample_goal(sim)

        ep_ret, success = 0.0, False

        for t in range(args.max_steps):
            global_step += 1

            # current obs
            rgb = bgr(sim.get_sensor_observations()["color_sensor"])
            s = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.
            with torch.no_grad():
                s_embed = encoder(s)

            action_idx = cvm_agent.act(s_embed) 
            action_str = ACTION_MAP[action_idx]
            sim.step(action_str)

            # next obs
            n_rgb = bgr(sim.get_sensor_observations()["color_sensor"])
            
            # CVM Bonus Calculation
            ns = torch.from_numpy(n_rgb).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.
            with torch.no_grad():
                ns_embed = encoder(ns)
                # Compute displacement
                delta_phi = ns_embed - s_embed
                
                # Compute bonus
                bonus = cvm_agent.cvm.get_bonus(delta_phi).item()
                
                # Update CVM estimator
                cvm_agent.cvm.update(delta_phi)

            dist = euclid(agent_sim.get_state().position, goal)
            reward, done = -dist, False
            
            # Add intrinsic bonus
            reward += args.cvm_coef * bonus
            
            if dist < args.goal_thresh:
                reward += 100.0
                done, success = True, True
            if t == args.max_steps - 1:
                reward -= 50.0
                done = True

            ep_ret += reward
            replay.push((rgb, np.array([action_idx]), reward, n_rgb, float(done)))

            if done:
                replay.finish_episode(success)
                break

            # periodic updates
            if global_step % args.update_freq == 0 and len(replay) >= args.warmup:
                rl_batch = replay.sample_rl(args.batch)
                gear_agent.update_rl(rl_batch, encoder, alpha=args.alpha)
                gear_agent.update_generators(replay, encoder,
                                             args.K_pos, args.K_neg)

        returns.append(ep_ret)
        if ep % args.log_every == 0:
            log(f"Ep {ep:4d} | Ret {ep_ret:7.1f} | Avg "
                f"{np.mean(returns[-args.log_every:]):7.1f} | Succ {success}")

        # checkpoint
        if ep % args.ckpt_every == 0 and ep > 0:
            torch.save({
                "agent": cvm_agent.state_dict(),
                "encoder": encoder.state_dict(),
                "step": global_step,
                "episode": ep,
            }, os.path.join(args.save_dir, f"ckpt_{ep}.pt"))

    sim.close()
    cv2.destroyAllWindows()

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, help="Path to HM3D .glb")
    p.add_argument("--save_dir", default="runs/gear")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--goal_thresh", type=float, default=0.5)
    p.add_argument("--state_dim", type=int, default=256)
    p.add_argument("--goal_dim", type=int, default=256)
    p.add_argument("--buffer_cap", type=int, default=100_000)
    p.add_argument("--t_max", type=int, default=50)
    p.add_argument("--update_freq", type=int, default=20)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--ckpt_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.2, help="Entropy temperature for SAC style updates")
    
    return p.parse_args()


if __name__ == "__main__":
    train(get_args())

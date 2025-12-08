import os
import time
import json
from collections import deque, namedtuple
import random

import cv2
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("results", exist_ok=True)

DQN_ENV_ID = "CarRacing-v2"
DQN_TOTAL_FRAMES = 300_000
DQN_REPLAY_CAPACITY = 100_000
DQN_BATCH_SIZE = 32
DQN_GAMMA = 0.99

DQN_EPS_START = 1.0
DQN_EPS_END = 0.1
DQN_EPS_DECAY_FRAMES = 250_000

DQN_LR = 1e-4
DQN_TARGET_UPDATE_FREQ = 1_000

DQN_Transition = namedtuple(
    "DQN_Transition", ("state", "action", "reward", "next_state", "done")
)


class DQNReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(DQN_Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN_QNetwork(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),  # -> (16, 20, 20)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # -> (32, 9, 9)
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),  # 32 * 9 * 9 = 2592
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


def dqn_preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """RGB (96,96,3) -> grayscale (84,84) in [0,1]"""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def dqn_stack_frames(frames_deque: deque) -> np.ndarray:
    """Deque of 4 frames (84,84) -> (4,84,84)"""
    assert len(frames_deque) == 4
    return np.stack(frames_deque, axis=0)


def dqn_epsilon_by_frame(frame_idx: int) -> float:
    if frame_idx >= DQN_EPS_DECAY_FRAMES:
        return DQN_EPS_END
    eps = DQN_EPS_START - (DQN_EPS_START - DQN_EPS_END) * (
        frame_idx / DQN_EPS_DECAY_FRAMES
    )
    return max(DQN_EPS_END, eps)


def dqn_optimize(policy_net, target_net, memory, optimizer):
    if len(memory) < DQN_BATCH_SIZE:
        return None

    transitions = memory.sample(DQN_BATCH_SIZE)
    batch = DQN_Transition(*zip(*transitions))

    state_batch = torch.from_numpy(np.stack(batch.state)).to(DEVICE)  # (B,4,84,84)
    action_batch = torch.tensor(
        batch.action, dtype=torch.long, device=DEVICE
    ).unsqueeze(
        1
    )  # (B,1)
    reward_batch = torch.tensor(
        batch.reward, dtype=torch.float32, device=DEVICE
    )  # (B,)
    next_state_batch = torch.from_numpy(np.stack(batch.next_state)).to(
        DEVICE
    )  # (B,4,84,84)
    done_batch = torch.tensor(batch.done, dtype=torch.float32, device=DEVICE)  # (B,)

    # Q(s,a)
    q_values = policy_net(state_batch).gather(1, action_batch).squeeze(1)

    # Target: r + γ * max_a' Q_target(s',a')
    with torch.no_grad():
        next_q_values = target_net(next_state_batch).max(1)[0]
        target_q = reward_batch + DQN_GAMMA * next_q_values * (1.0 - done_batch)

    criterion = nn.SmoothL1Loss()
    loss = criterion(q_values, target_q)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_value_(policy_net.parameters(), 1.0)
    optimizer.step()

    return loss.item()


def train_dqn():
    run_name = "dqn_carracing1"
    logdir = os.path.join("runs", run_name)
    os.makedirs("results/DQN", exist_ok=True)

    writer = SummaryWriter(log_dir=logdir)
    print(f"[DQN] TensorBoard logdir: {logdir}")

    env = gym.make(DQN_ENV_ID, render_mode=None, continuous=False)
    num_actions = env.action_space.n

    policy_net = DQN_QNetwork(num_actions).to(DEVICE)
    target_net = DQN_QNetwork(num_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=DQN_LR)
    memory = DQNReplayMemory(DQN_REPLAY_CAPACITY)

    global_frame = 0
    episode_idx = 0

    episode_rewards = []
    loss_history = []

    pbar = tqdm(total=DQN_TOTAL_FRAMES, desc="DQN Training", dynamic_ncols=True)

    start_time = time.time()

    while global_frame < DQN_TOTAL_FRAMES:
        frame, info = env.reset()
        frame_p = dqn_preprocess_frame(frame)
        frames = deque([frame_p] * 4, maxlen=4)
        state = dqn_stack_frames(frames)

        episode_reward = 0.0
        episode_steps = 0
        done = False

        while not done and global_frame < DQN_TOTAL_FRAMES:
            global_frame += 1
            episode_steps += 1
            pbar.update(1)

            eps = dqn_epsilon_by_frame(global_frame)

            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s = torch.from_numpy(state).unsqueeze(0).to(DEVICE)
                    q_values = policy_net(s)
                    action = q_values.argmax(dim=1).item()

            next_frame, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            reward_clipped = float(np.clip(reward, -1.0, 1.0))
            episode_reward += reward

            next_frame_p = dqn_preprocess_frame(next_frame)
            frames.append(next_frame_p)
            next_state = dqn_stack_frames(frames)

            memory.push(state, action, reward_clipped, next_state, done)

            state = next_state

            loss_val = dqn_optimize(policy_net, target_net, memory, optimizer)
            if loss_val is not None:
                loss_history.append(loss_val)

            if loss_val is not None and global_frame % 100 == 0:
                writer.add_scalar("loss", loss_val, global_frame)

            if global_frame % DQN_TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if global_frame % 1000 == 0:
                writer.add_scalar("epsilon", eps, global_frame)

            if episode_steps >= 1000:
                break

        episode_idx += 1
        episode_rewards.append(episode_reward)
        writer.add_scalar("episode_reward", episode_reward, global_frame)

        print(
            f"[DQN] Episode {episode_idx} | Frames {global_frame} | "
            f"Reward {episode_reward:.2f} | eps={eps:.3f}"
        )

    pbar.close()
    env.close()
    writer.close()

    torch.save(policy_net.state_dict(), "results/DQN/dqn_carracing_final.pth")
    print("[DQN] Training done. Model saved to results/DQN/dqn_carracing_final.pth")
    print(f"[DQN] Total time: {(time.time() - start_time)/60:.1f} min")

    return {
        "episode_rewards": episode_rewards,
        "losses": loss_history,
    }


PPO_ENV_ID = "CarRacing-v2"
PPO_TOTAL_TIMESTEPS = 300_000
PPO_ROLLOUT_STEPS = 2_048
PPO_NUM_EPOCHS = 10
PPO_MINIBATCH_SIZE = 256
PPO_GAMMA = 0.99
PPO_LAMBDA = 0.95
PPO_CLIP_EPS = 0.2
PPO_LR = 3e-4
PPO_ENT_COEF = 0.02
PPO_VF_COEF = 0.5
PPO_MAX_GRAD_NORM = 0.5
PPO_FRAME_STACK_K = 4


class PPO_GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space
        h, w, _ = obs_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(h, w, 1),
            dtype=np.uint8,
        )

    def observation(self, obs):
        gray = obs.mean(axis=2, keepdims=True)
        return gray.astype(np.uint8)


class PPO_SimpleFrameStack(gym.Wrapper):
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)

        orig_space = env.observation_space
        H, W, C = orig_space.shape
        assert C == 1, "GrayScaleObservation should give C=1"

        low = np.repeat(orig_space.low, k, axis=2)
        high = np.repeat(orig_space.high, k, axis=2)
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=orig_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=2)


class PPO_ActorCriticCNN(nn.Module):
    def __init__(self, input_channels, action_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 84, 84)
            conv_out = self.conv(dummy)
            conv_out_dim = conv_out.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim, 256),
            nn.ReLU(),
        )

        self.mu_head = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.v_head = nn.Linear(256, 1)

    def forward(self, x):
        # x: (B,H,W,C) -> (B,C,H,W)
        x = x.permute(0, 3, 1, 2)
        x = x / 255.0
        x = nn.functional.interpolate(
            x, size=(84, 84), mode="bilinear", align_corners=False
        )
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def policy(self, x):
        z = self.forward(x)
        mu = torch.tanh(self.mu_head(z))
        std = torch.exp(self.log_std)
        return mu, std

    def value(self, x):
        z = self.forward(x)
        v = self.v_head(z)
        return v.squeeze(-1)

    def act(self, x):
        mu, std = self.policy(x)
        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob

    def evaluate_actions(self, x, actions):
        mu, std = self.policy(x)
        dist = Normal(mu, std)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        values = self.value(x)
        return log_probs, entropy, values


class PPO_RolloutBuffer:
    def __init__(self, size, obs_shape, action_dim):
        self.size = size
        self.ptr = 0

        self.obs = np.zeros((size,) + obs_shape, dtype=np.uint8)
        self.actions = np.zeros((size, action_dim), dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.bool_)
        self.values = np.zeros(size, dtype=np.float32)

        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)

    def add(self, obs, action, log_prob, reward, done, value):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_returns_and_advantages(self, last_value, gamma, lam):
        adv = 0.0
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_non_terminal = 1.0 - float(self.dones[t])
                next_value = last_value
            else:
                next_non_terminal = 1.0 - float(self.dones[t + 1])
                next_value = self.values[t + 1]

            delta = (
                self.rewards[t]
                + gamma * next_value * next_non_terminal
                - self.values[t]
            )
            adv = delta + gamma * lam * next_non_terminal * adv
            self.advantages[t] = adv

        self.returns = self.advantages + self.values
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std() + 1e-8
        self.advantages = (self.advantages - adv_mean) / adv_std

    def get_tensors(self, device):
        return (
            torch.from_numpy(self.obs).float().to(device),
            torch.from_numpy(self.actions).float().to(device),
            torch.from_numpy(self.log_probs).float().to(device),
            torch.from_numpy(self.advantages).float().to(device),
            torch.from_numpy(self.returns).float().to(device),
        )


def train_ppo():
    run_name = f"ppo_carracing1"
    logdir = os.path.join("runs", run_name)
    os.makedirs("results/PPO", exist_ok=True)

    writer = SummaryWriter(log_dir=logdir)
    print(f"[PPO] TensorBoard logdir: {logdir}")

    base_env = gym.make(PPO_ENV_ID, render_mode=None)
    gray_env = PPO_GrayScaleObservation(base_env)
    env = PPO_SimpleFrameStack(gray_env, k=PPO_FRAME_STACK_K)

    obs, info = env.reset()
    obs_shape = env.observation_space.shape
    action_space = env.action_space

    action_dim = action_space.shape[0]
    input_channels = obs_shape[2]

    model = PPO_ActorCriticCNN(input_channels, action_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=PPO_LR)

    global_step = 0
    episode_rewards = deque(maxlen=10)
    all_episode_rewards = []
    value_losses = []
    policy_losses = []
    entropies = []

    episode_reward = 0.0

    pbar = tqdm(total=PPO_TOTAL_TIMESTEPS, desc="PPO Training", dynamic_ncols=True)
    start_time = time.time()

    while global_step < PPO_TOTAL_TIMESTEPS:
        buffer = PPO_RolloutBuffer(PPO_ROLLOUT_STEPS, obs_shape, action_dim)

        # Collect rollouts
        for step in range(PPO_ROLLOUT_STEPS):
            global_step += 1

            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                value = model.value(obs_tensor).cpu().item()
                action, log_prob = model.act(obs_tensor)
            action = action.cpu().numpy()[0]

            action_clipped = np.clip(action, -1.0, 1.0)
            action_clipped[0] *= 0.7
            action_clipped[1] = np.clip(action_clipped[1], 0.0, 1.0)
            action_clipped[2] = np.clip(action_clipped[2], 0.0, 1.0)

            next_obs, reward, terminated, truncated, info = env.step(action_clipped)
            done = terminated or truncated

            shaped_reward = reward
            if abs(action_clipped[0]) < 0.1:
                shaped_reward += 0.02
            if action_clipped[1] > 0.3:
                shaped_reward += 0.02
            if action_clipped[2] > 0.1:
                shaped_reward -= 0.02

            scaled_reward = shaped_reward / 10.0

            buffer.add(
                obs=obs,
                action=action_clipped,
                log_prob=log_prob.cpu().item(),
                reward=scaled_reward,
                done=done,
                value=value,
            )

            obs = next_obs
            episode_reward += reward

            if done:
                episode_rewards.append(episode_reward)
                all_episode_rewards.append(episode_reward)
                writer.add_scalar("charts/episode_reward", episode_reward, global_step)
                obs, info = env.reset()
                episode_reward = 0.0

            if global_step >= PPO_TOTAL_TIMESTEPS:
                break

        pbar.update(PPO_ROLLOUT_STEPS)

        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            last_value = model.value(obs_tensor).cpu().item()

        buffer.compute_returns_and_advantages(last_value, PPO_GAMMA, PPO_LAMBDA)

        (
            obs_batch,
            actions_batch,
            old_log_probs_batch,
            advantages_batch,
            returns_batch,
        ) = buffer.get_tensors(DEVICE)

        batch_size = PPO_ROLLOUT_STEPS
        indices = np.arange(batch_size)

        for epoch in trange(
            PPO_NUM_EPOCHS, desc="PPO Update", leave=False, dynamic_ncols=True
        ):
            np.random.shuffle(indices)

            for start in range(0, batch_size, PPO_MINIBATCH_SIZE):
                end = start + PPO_MINIBATCH_SIZE
                mb_idx = indices[start:end]

                mb_obs = obs_batch[mb_idx]
                mb_actions = actions_batch[mb_idx]
                mb_old_log_probs = old_log_probs_batch[mb_idx]
                mb_advantages = advantages_batch[mb_idx]
                mb_returns = returns_batch[mb_idx]

                new_log_probs, entropy, values = model.evaluate_actions(
                    mb_obs, mb_actions
                )
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS)
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = ((values - mb_returns) ** 2).mean()
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss + PPO_VF_COEF * value_loss + PPO_ENT_COEF * entropy_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), PPO_MAX_GRAD_NORM)
                optimizer.step()

        if len(episode_rewards) > 0:
            writer.add_scalar(
                "charts/mean_episode_reward", np.mean(episode_rewards), global_step
            )

        writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)

        value_losses.append(value_loss.item())
        policy_losses.append(policy_loss.item())
        entropies.append(entropy.mean().item())

        print(
            f"[PPO] Step: {global_step} | "
            f"MeanEpRew (last 10): {np.mean(episode_rewards) if episode_rewards else 0:.2f}"
        )

    pbar.close()
    env.close()
    writer.close()
    torch.save(model.state_dict(), "results/PPO/ppo_carracing_from_scratch.pth")
    print(
        "[PPO] Training finished, model saved to results/PPO/ppo_carracing_from_scratch.pth"
    )
    print(f"[PPO] Total time: {(time.time() - start_time)/60:.1f} min")

    return {
        "episode_rewards": all_episode_rewards,
        "value_losses": value_losses,
        "policy_losses": policy_losses,
        "entropies": entropies,
    }


def plot_benchmark(results):
    plt.figure(figsize=(10, 5))
    for algo, data in results.items():
        plt.plot(data["episode_rewards"], label=algo)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode Reward vs Algorithm")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/benchmark_rewards.png")
    plt.close()

    if "losses" in results.get("DQN", {}):
        plt.figure(figsize=(10, 5))
        plt.plot(results["DQN"]["losses"])
        plt.xlabel("Optimization Step")
        plt.ylabel("DQN Loss")
        plt.title("DQN Loss Curve")
        plt.grid(True)
        plt.savefig("results/dqn_loss_curve.png")
        plt.close()

    if "value_losses" in results.get("PPO", {}):
        plt.figure(figsize=(10, 5))
        plt.plot(results["PPO"]["value_losses"], label="Value loss")
        plt.plot(results["PPO"]["policy_losses"], label="Policy loss")
        plt.xlabel("Update")
        plt.ylabel("Loss")
        plt.title("PPO Loss Curves")
        plt.legend()
        plt.grid(True)
        plt.savefig("results/ppo_loss_curves.png")
        plt.close()


def benchmark():
    big_results = {}
    dqn_hist = train_dqn()
    dqn_rewards = dqn_hist["episode_rewards"]
    big_results["DQN"] = {
        **dqn_hist,
        "best_reward": max(dqn_rewards) if dqn_rewards else None,
        "mean_last_10": np.mean(dqn_rewards[-10:]) if len(dqn_rewards) >= 10 else None,
    }

    ppo_hist = train_ppo()
    ppo_rewards = ppo_hist["episode_rewards"]
    big_results["PPO"] = {
        **ppo_hist,
        "best_reward": max(ppo_rewards) if ppo_rewards else None,
        "mean_last_10": np.mean(ppo_rewards[-10:]) if len(ppo_rewards) >= 10 else None,
    }

    summary = {
        algo: {
            "best_reward": res["best_reward"],
            "mean_last_10": res["mean_last_10"],
            "num_episodes": len(res["episode_rewards"]),
        }
        for algo, res in big_results.items()
    }

    with open("results/benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_benchmark(big_results)
    print("Testing finished")


if __name__ == "__main__":
    benchmark()

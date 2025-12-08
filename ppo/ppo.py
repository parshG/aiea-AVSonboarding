import os
import time
from collections import deque
from tqdm import tqdm, trange

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

ENV_ID = "CarRacing-v2"
TOTAL_TIMESTEPS = 500000
ROLLOUT_STEPS = 2048
NUM_EPOCHS = 10
MINIBATCH_SIZE = 256
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 3e-4
ENT_COEF = 0.02
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
FRAME_STACK_K = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space
        assert len(obs_space.shape) == 3, "Expected image observations"
        h, w, _ = obs_space.shape

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(h, w, 1),
            dtype=np.uint8,
        )

    def observation(self, obs):
        # obs: (H, W, 3), convert to (H, W, 1)
        gray = obs.mean(axis=2, keepdims=True)
        return gray.astype(np.uint8)


class SimpleFrameStack(gym.Wrapper):
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)

        orig_space = env.observation_space
        assert len(orig_space.shape) == 3, "Expected image observations"
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


class ActorCriticCNN(nn.Module):
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
        mu = self.mu_head(z)
        mu = torch.tanh(mu)
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


class RolloutBuffer:
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


def train():
    run_name = f"ppo_carracing_from_scratch_{int(time.time())}"
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))

    base_env = gym.make(ENV_ID, render_mode=None)
    gray_env = GrayScaleObservation(base_env)
    env = SimpleFrameStack(gray_env, k=FRAME_STACK_K)

    obs, info = env.reset()
    obs_shape = env.observation_space.shape
    action_space = env.action_space

    assert len(obs_shape) == 3, "Expected image observations"
    assert len(action_space.shape) == 1, "Expected continuous actions"

    action_dim = action_space.shape[0]
    input_channels = obs_shape[2]

    model = ActorCriticCNN(input_channels, action_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    global_step = 0
    episode_rewards = deque(maxlen=10)
    episode_reward = 0.0
    pbar = tqdm(total=TOTAL_TIMESTEPS, desc="Training PPO", dynamic_ncols=True)

    while global_step < TOTAL_TIMESTEPS:
        buffer = RolloutBuffer(ROLLOUT_STEPS, obs_shape, action_dim)
        for step in range(ROLLOUT_STEPS):
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
                writer.add_scalar("charts/episode_reward", episode_reward, global_step)
                obs, info = env.reset()
                episode_reward = 0.0

        pbar.update(ROLLOUT_STEPS)

        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            last_value = model.value(obs_tensor).cpu().item()

        buffer.compute_returns_and_advantages(last_value, GAMMA, LAMBDA)

        (
            obs_batch,
            actions_batch,
            old_log_probs_batch,
            advantages_batch,
            returns_batch,
        ) = buffer.get_tensors(DEVICE)

        batch_size = ROLLOUT_STEPS
        indices = np.arange(batch_size)

        for epoch in trange(
            NUM_EPOCHS, desc="PPO Update", leave=False, dynamic_ncols=True
        ):
            np.random.shuffle(indices)

            for start in range(0, batch_size, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
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
                    torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = ((values - mb_returns) ** 2).mean()
                entropy_loss = -entropy.mean()

                loss = policy_loss + VF_COEF * value_loss + ENT_COEF * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        if len(episode_rewards) > 0:
            writer.add_scalar(
                "charts/mean_episode_reward", np.mean(episode_rewards), global_step
            )
        writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)

        print(
            f"Step: {global_step} | MeanEpRew: {np.mean(episode_rewards) if episode_rewards else 0:.2f}"
        )

    pbar.close()
    env.close()
    writer.close()
    torch.save(model.state_dict(), "ppo_carracing_from_scratch.pth")
    print("Training finished")


if __name__ == "__main__":
    train()

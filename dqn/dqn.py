import os
import time
from collections import deque, namedtuple
import random

import cv2
import gymnasium as gym
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

ENV_ID = "CarRacing-v2"
TOTAL_FRAMES = 100000        # total env timesteps
REPLAY_CAPACITY = 100_000
BATCH_SIZE = 32
GAMMA = 0.99

EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY_FRAMES = 250_000     # frames over which epsilon decays

LR = 1e-4                      # learning rate
TARGET_UPDATE_FREQ = 10_000    # how often (in frames) to copy weights to target net

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple(
    "Transition",
    ("state", "action", "reward", "next_state", "done")
)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """Save a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class QNetwork(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),  
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),                
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0)


def stack_frames(frames_deque: deque) -> np.ndarray:
    assert len(frames_deque) == 4
    return np.stack(frames_deque, axis=0)

def epsilon_by_frame(frame_idx: int) -> float:
    if frame_idx >= EPS_DECAY_FRAMES:
        return EPS_END
    eps = EPS_START - (EPS_START - EPS_END) * (frame_idx / EPS_DECAY_FRAMES)
    return max(EPS_END, eps)

def train():
    run_name = f"dqn_carracing_{int(time.time())}"
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))
    print(f"TensorBoard logdir: runs/{run_name}")

    env = gym.make(ENV_ID, render_mode=None, continuous=False)
    num_actions = env.action_space.n  

    policy_net = QNetwork(num_actions).to(DEVICE)
    target_net = QNetwork(num_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(REPLAY_CAPACITY)

    global_frame = 0
    episode_idx = 0
    best_mean_reward = -float("inf")

    pbar = tqdm(total=TOTAL_FRAMES, desc="DQN Training", dynamic_ncols=True)

    while global_frame < TOTAL_FRAMES:
        frame, info = env.reset()
        frame_p = preprocess_frame(frame)
        frames = deque([frame_p] * 4, maxlen=4)
        state = stack_frames(frames)   
        episode_reward = 0.0
        episode_steps = 0

        done = False
        while not done and global_frame < TOTAL_FRAMES:
            global_frame += 1
            episode_steps += 1
            pbar.update(1)

            eps = epsilon_by_frame(global_frame)

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

            next_frame_p = preprocess_frame(next_frame)
            frames.append(next_frame_p)
            next_state = stack_frames(frames)  

            memory.push(
                state,
                action,
                reward_clipped,
                next_state,
                done,
            )

            state = next_state

            loss_val = optimize_model(
                policy_net,
                target_net,
                memory,
                optimizer,
            )

            if loss_val is not None and global_frame % 100 == 0:
                writer.add_scalar("loss", loss_val, global_frame)

            if global_frame % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if global_frame % 1000 == 0:
                writer.add_scalar("epsilon", eps, global_frame)

            if episode_steps >= 1000:
                break

        episode_idx += 1
        writer.add_scalar("episode_reward", episode_reward, global_frame)
        print(f"Episode {episode_idx} | Frames {global_frame} | Reward {episode_reward:.2f} | eps={eps:.3f}")

        episode_rewards = np.array(
            [event.value for event in writer._get_file_writer()._summary_writer._event_writer._ev_writer._queue]  # ugly, ignore; use explicit tracking if you want
        ) if False else None 

    pbar.close()
    env.close()
    writer.close()

    torch.save(policy_net.state_dict(), "dqn_carracing_final.pth")
    print("Training done")


def optimize_model(policy_net, target_net, memory, optimizer):
    if len(memory) < BATCH_SIZE:
        return None

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.from_numpy(np.stack(batch.state)).to(DEVICE)    
    action_batch = torch.tensor(batch.action, dtype=torch.long, device=DEVICE).unsqueeze(1)  
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=DEVICE)            
    next_state_batch = torch.from_numpy(np.stack(batch.next_state)).to(DEVICE)         
    done_batch = torch.tensor(batch.done, dtype=torch.float32, device=DEVICE)            

    q_values = policy_net(state_batch).gather(1, action_batch).squeeze(1)  

    with torch.no_grad():
        next_q_values = target_net(next_state_batch).max(1)[0]            
        target_q = reward_batch + GAMMA * next_q_values * (1.0 - done_batch)

    criterion = nn.SmoothL1Loss()
    loss = criterion(q_values, target_q)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_value_(policy_net.parameters(), 1.0)
    optimizer.step()

    return loss.item()

if __name__ == "__main__":
    train()

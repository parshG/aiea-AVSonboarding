import gymnasium as gym 
from stable_baselines3 import PPO 

model = PPO.load("ppo_carRacing_500k")

env = gym.make("CarRacing-v2", render_mode="human")

obs, info = env.reset()

print(f"Starting observation: {obs}")

episode_over = False
total_reward = 0

while not episode_over:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, info = env.step(action)
    total_reward += rewards
    episode_over = terminated or truncated 

print(f"Episode finished! Total reward: {total_reward}")
env.close()
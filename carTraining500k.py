import gymnasium as gym 
from stable_baselines3 import PPO

env = gym.make("CarRacing-v2", render_mode = "rgb_array")

model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./carRacingTensorboard")
model.learn(total_timesteps=500000, progress_bar=True)
model.save("ppo_carRacing_500k")

env.close()
import gym
import gym_gomoku
from stable_baselines3 import PPO
from huggingface_hub import hf_hub_download
# from huggingface_hub import login, upload_file


pretrained_model_repo = "sb3/ppo-CartPole-v1"
pretrained_model_filename = "ppo-CartPole-v1.zip"
reinforced_model_filename = "gomoku_ppo"


# Load pre-trained PPO model
model_path = hf_hub_download(repo_id=pretrained_model_repo, filename=pretrained_model_filename)
model = PPO.load(model_path)

# Set up Gomoku environment
env = gym.make('Gomoku19x19-v0', disable_env_checker=True)

# Fine-tune the model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save(reinforced_model_name)

# Upload to Hugging Face Hub
# login()  # Hugging Face token
# upload_file(path_or_fileobj="gomoku_ppo.zip", path_in_repo="gomoku_ppo.zip", repo_id="your-username/gomoku-ppo")

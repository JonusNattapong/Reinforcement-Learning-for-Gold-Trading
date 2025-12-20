import os
from safetensors.torch import save_file
from stable_baselines3 import PPO

# Load the trained PPO model
model_path = os.path.join("models", "ppo_xauusd.zip")
model = PPO.load(model_path)

# Extract the policy state dict
state_dict = model.policy.state_dict()

# Save as safetensors
output_path = os.path.join("models", "ppo_xauusd.safetensors")
save_file(state_dict, output_path)

print(f"Model exported to {output_path}")
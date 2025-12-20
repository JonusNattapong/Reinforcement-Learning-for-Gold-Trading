import os
from huggingface_hub import HfApi

# Get HF token from environment or user input
token = os.environ.get("HF_TOKEN")
if not token:
    token = input("Enter your Hugging Face token: ")

# Initialize API
api = HfApi(token=token)

# Define repo name (adjust as needed)
repo_name = "JonusNattapong/Reinforcement-Learning-for-Gold-Trading-Model"

# Create repo if it doesn't exist
api.create_repo(repo_name, exist_ok=True, private=False)

# Upload safetensors model
model_path = os.path.join("models", "ppo_xauusd.safetensors")
api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="ppo_xauusd.safetensors",
    repo_id=repo_name,
    commit_message="Upload PPO model for XAUUSD trading"
)

# Optionally upload VecNormalize stats
vec_path = os.path.join("models", "vecnormalize.pkl")
if os.path.exists(vec_path):
    api.upload_file(
        path_or_fileobj=vec_path,
        path_in_repo="vecnormalize.pkl",
        repo_id=repo_name,
        commit_message="Upload VecNormalize stats"
    )

print(f"Model uploaded to https://huggingface.co/{repo_name}")
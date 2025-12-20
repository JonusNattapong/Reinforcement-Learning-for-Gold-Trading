import os
from huggingface_hub import HfApi

# Get HF token
token = os.environ.get("HF_TOKEN")
if not token:
    token = input("Enter your Hugging Face token: ")

# Initialize API
api = HfApi(token=token)

# Repo name
repo_name = "JonusNattapong/Reinforcement-Learning-for-Gold-Trading-Model"

# Upload README
readme_path = "README_HF.md"
api.upload_file(
    path_or_fileobj=readme_path,
    path_in_repo="README.md",
    repo_id=repo_name,
    commit_message="Add comprehensive README with model details, metrics, and usage instructions"
)

print(f"README uploaded to https://huggingface.co/{repo_name}")
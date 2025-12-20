---
license: mit
language: en
library_name: stable-baselines3
tags:
- reinforcement-learning
- finance
- gold-trading
- xauusd
- ppo
metrics:
- sharpe_ratio
- win_rate
pipeline_tag: reinforcement-learning
---

# PPO Model for XAUUSD Gold Trading

This repository contains a Reinforcement Learning model trained using Proximal Policy Optimization (PPO) for trading XAUUSD (Gold vs US Dollar) on 15-minute timeframes.

## Model Details

- **Model Type**: PPO (Proximal Policy Optimization)
- **Framework**: Stable-Baselines3
- **Environment**: Custom Gym environment for XAUUSD trading
- **Training Data**: Historical XAUUSD data from 2004 to 2025 (resampled to 15-min bars)
- **Total Timesteps**: 1,000,000
- **Position Sizing**: Base 5.0 oz, Max 7.5 oz
- **Initial Capital**: 200 USD
- **Transaction Cost**: 0.65 USD per oz

## Performance Metrics (Test Set)

- **Average Daily Profit**: 51.46 USD
- **Win Rate**: 69.0%
- **Max Drawdown**: 12.0%
- **Sharpe Ratio**: 7.56
- **Average Trades per Day**: 2.66

## Features Used

- Log Return
- RSI (14-period)
- Moving Averages (short/long)
- Bollinger Bands
- MACD
- Volume indicators

## Usage

### Loading the Model

Below are two safe ways to load the trained policy depending on what you have available.

Option A — Load the full Stable-Baselines3 model (.zip)

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
import os

# Create or reconstruct an environment similar to the one used for training
# e.g. `env = make_your_env(...)` — replace with your env factory
env = ...

# If you saved VecNormalize separately, load and wrap your env first
if os.path.exists("models/vecnormalize.pkl"):
	vec = VecNormalize.load("models/vecnormalize.pkl", env)
	vec.training = False
	vec.norm_reward = False
	env = vec

# Load the full model (policy + optimizer state)
model = PPO.load("models/ppo_xauusd.zip", env=env)
```

Option B — Load weights saved as SafeTensors into a fresh PPO policy

```python
from safetensors.torch import load_file
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
import os

# Create or reconstruct the same environment used for training
env = ...

# If you have VecNormalize statistics, load them and wrap the env
if os.path.exists("models/vecnormalize.pkl"):
	vec = VecNormalize.load("models/vecnormalize.pkl", env)
	vec.training = False
	vec.norm_reward = False
	env = vec

# Instantiate a PPO model with the same policy architecture
model = PPO("MlpPolicy", env)

# Load SafeTensors state dict and convert values to torch.Tensor if needed
raw_state = load_file("models/ppo_xauusd.safetensors")
state_dict = {k: (torch.tensor(v) if not isinstance(v, torch.Tensor) else v) for k, v in raw_state.items()}

# Load weights into the policy
model.policy.load_state_dict(state_dict)

# Ensure the model has the same env wrapper
model.set_env(env)
```

Notes:
- Option A is preferred when `ppo_xauusd.zip` is available (it contains the entire SB3 model).
- Option B is useful when only the policy weights were exported as SafeTensors. Ensure the policy architecture and observation/action spaces match the original training setup.
- Always set `vec.training = False` and `vec.norm_reward = False` when running inference.


### For Full Inference

To use the model for trading, you'll need to:
1. Set up the trading environment (`XAUUSDTradingEnv`)
2. Load VecNormalize stats
3. Run predictions

Note: This is a simulation model. Use with caution in real trading.

## Training Configuration

- Learning Rate: 0.0003
- Batch Size: 256
- Gamma: 0.99
- GAE Lambda: 0.95
- Clip Range: 0.2
- Entropy Coefficient: 0.01

## Files

- `ppo_xauusd.safetensors`: Model weights in SafeTensors format
- `vecnormalize.pkl`: VecNormalize statistics for observation normalization

## License

MIT License

## Disclaimer

This model is for educational and research purposes only. Trading involves risk, and past performance does not guarantee future results. Always backtest and validate before using in live trading.
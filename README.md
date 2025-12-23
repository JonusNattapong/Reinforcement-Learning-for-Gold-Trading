# Reinforcement Learning for Gold Trading

This repository implements a Reinforcement Learning (RL) system for trading XAUUSD (Gold vs US Dollar) using Proximal Policy Optimization (PPO). The model is trained on historical gold price data from 2004 to 2025, resampled to 15-minute intervals.

## Features

- **Custom Trading Environment**: Gymnasium-based environment simulating gold trading with realistic constraints
- **Advanced Features**: Log returns, RSI, Moving Averages, Bollinger Bands, MACD, and volume indicators
- **Risk Management**: Position sizing, daily loss limits, profit targets, and drawdown controls
- **Evaluation Metrics**: Sharpe ratio, win rate, max drawdown, and daily profit analysis
- **Pre-trained Model**: Ready-to-use PPO model trained on 1 million timesteps

## Model Performance (Test Set: 2024-2025)

- **Average Daily Profit**: $51.46
- **Win Rate**: 69.0%
- **Max Drawdown**: 12.0%
- **Sharpe Ratio**: 7.56
- **Average Trades per Day**: 2.66

## Installation

1. Clone the repository:
```bash
git clone https://github.com/JonusNattapong/Reinforcement-Learning-for-Gold-Trading.git
cd Reinforcement-Learning-for-Gold-Trading
```

2. Create a virtual environment:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Unix/MacOS:
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install stable-baselines3 gymnasium pandas numpy datasets safetensors torch
```

## Usage

### Training a New Model

To train a new PPO model:

```bash
python train.py --mode train --timesteps 1000000
```

Optional arguments:
- `--csv`: Path to custom CSV data file (falls back to Hugging Face dataset)
- `--timesteps`: Total training timesteps (default: 1,000,000)
- `--save-dir`: Directory to save model (default: models)

### Evaluating the Pre-trained Model

To evaluate the existing model on test data:

```bash
python train.py --mode eval
```

### Training and Evaluating

To train a new model and evaluate it:

```bash
python train.py --mode train_eval
```

## Configuration

The system uses three main configuration classes:

- **DataConfig**: Data loading and splitting parameters
- **EnvConfig**: Trading environment settings (capital, position sizes, costs, penalties)
- **TrainConfig**: PPO training hyperparameters

Modify these in `src/rl_gold_trading/config.py` to customize behavior.

## Data

The model uses XAUUSD historical data. By default, it loads from the Hugging Face dataset `ZombitX64/xauusd-gold-price-historical-data-2004-2025`. You can provide your own CSV file with columns: datetime, open, high, low, close, volume.

## Loading the Trained Model

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from safetensors.torch import load_file
import torch

# Load the SafeTensors model
model = PPO("MlpPolicy", env)  # env should match training environment
state_dict = load_file("models/ppo_xauusd.safetensors")
model.policy.load_state_dict(state_dict)
```

## Project Structure

```
├── src/rl_gold_trading/
│   ├── __init__.py
│   ├── config.py          # Configuration dataclasses
│   ├── data.py            # Data loading and preprocessing
│   ├── envs.py            # Gymnasium trading environment
│   ├── features.py        # Technical indicators
│   ├── metrics.py         # Evaluation metrics
│   ├── run.py             # Main training/evaluation script
│   ├── train.py           # PPO training logic
│   └── vec_env.py         # Vectorized environment utilities
├── models/                # Saved models and normalization stats
├── train.py               # Entry point script
└── README.md
```

## Dependencies

- stable-baselines3
- gymnasium
- pandas
- numpy
- datasets (for Hugging Face data loading)
- safetensors
- torch

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Disclaimer

This is a research project for educational purposes. Trading involves risk and past performance does not guarantee future results. Always do your own research and consider consulting financial advisors before making trading decisions.


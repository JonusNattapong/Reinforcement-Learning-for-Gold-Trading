from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    csv_path: Optional[str] = None
    resample_rule: str = "15min"
    train_start_year: int = 2004
    train_end_year: int = 2021
    valid_start_year: int = 2022
    valid_end_year: int = 2023
    test_start_year: int = 2024
    test_end_year: int = 2025


@dataclass
class EnvConfig:
    initial_capital: float = 200.0
    base_size_oz: float = 5.0
    max_size_oz: float = 7.5
    transaction_cost_per_oz: float = 0.65
    daily_loss_limit: float = 50.0
    max_trades_per_day: int = 20
    soft_trade_limit: int = 15
    profit_target: float = 400.0
    drawdown_target: float = 0.15
    flip_penalty: float = 0.15
    drawdown_penalty: float = 2.5
    loss_limit_penalty: float = 75.0
    overtrade_penalty: float = 5.0
    end_day_bonus: float = 5.0
    min_bars_per_day: int = 60


@dataclass
class TrainConfig:
    timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    save_dir: str = "models"

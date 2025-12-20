import math
from typing import Dict

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from rl_gold_trading.vec_env import get_unwrapped_env, vec_reset, vec_step


def evaluate_model(
    model: PPO,
    vec_env: VecNormalize,
    day_count: int,
    initial_capital: float,
) -> Dict[str, float]:
    daily_pnls = []
    trade_counts = []
    base_env = get_unwrapped_env(vec_env)
    for day_idx in range(day_count):
        base_env.set_day_index(day_idx)
        obs, _ = vec_reset(vec_env)
        done = np.array([False])
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_step(vec_env, action)
        info0 = info[0]
        daily_pnls.append(info0["daily_pnl"])
        trade_counts.append(info0["trades"])

    daily_pnls = np.array(daily_pnls, dtype=np.float32)
    trade_counts = np.array(trade_counts, dtype=np.float32)
    equity_curve = initial_capital + np.cumsum(daily_pnls)
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (peaks - equity_curve) / np.maximum(peaks, 1e-6)
    sharpe = 0.0
    if daily_pnls.std() > 1e-6:
        sharpe = (daily_pnls.mean() / daily_pnls.std()) * math.sqrt(252)
    return {
        "avg_daily_profit": float(daily_pnls.mean()),
        "win_rate": float((daily_pnls > 0).mean()),
        "max_drawdown": float(drawdowns.max()),
        "sharpe": float(sharpe),
        "avg_trades_per_day": float(trade_counts.mean()),
    }

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from rl_gold_trading.config import EnvConfig


class XAUUSDTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        config: EnvConfig,
        random_reset: bool = True,
    ) -> None:
        super().__init__()
        self.df = df.reset_index()
        self.feature_cols = feature_cols
        self.config = config
        self.random_reset = random_reset
        self.prices = self.df["close"].to_numpy(dtype=np.float32)
        self.features = self.df[feature_cols].to_numpy(dtype=np.float32)
        self.timestamps = self.df["datetime"].to_numpy()
        self.day_indices = self._build_day_indices()
        self.day_cursor = 0
        self.forced_day_index: Optional[int] = None

        obs_dim = self.features.shape[1] + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self._reset_episode_state()

    def _build_day_indices(self) -> List[np.ndarray]:
        dates = pd.to_datetime(self.timestamps).date
        unique_days = pd.unique(dates)
        indices = []
        for day in unique_days:
            day_idx = np.where(dates == day)[0]
            if len(day_idx) >= self.config.min_bars_per_day:
                indices.append(day_idx)
        if not indices:
            raise ValueError("No days found with enough bars after preprocessing.")
        return indices

    def _reset_episode_state(self) -> None:
        self.position = 0
        self.position_size_oz = 0.0
        self.entry_price = np.nan
        self.equity = self.config.initial_capital
        self.start_equity = self.config.initial_capital
        self.peak_equity = self.config.initial_capital
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.prev_price = None

    def _select_day(self, day_index: Optional[int]) -> np.ndarray:
        if day_index is not None:
            return self.day_indices[day_index]
        if self.random_reset:
            return self.day_indices[np.random.randint(0, len(self.day_indices))]
        day = self.day_indices[self.day_cursor]
        self.day_cursor = (self.day_cursor + 1) % len(self.day_indices)
        return day

    def _position_size(self) -> float:
        scale = max(1.0, self.equity / self.config.initial_capital)
        size = self.config.base_size_oz * scale
        return float(np.clip(size, self.config.base_size_oz, self.config.max_size_oz))

    def _get_obs(self, row_idx: int) -> np.ndarray:
        feat = self.features[row_idx]
        unrealized = 0.0
        if self.position != 0 and not np.isnan(self.entry_price):
            unrealized = (
                (self.prices[row_idx] - self.entry_price)
                * self.position
                * self.position_size_oz
            )
        obs = np.concatenate(
            [
                feat,
                np.array(
                    [
                        self.position,
                        unrealized / self.config.initial_capital,
                        self.daily_pnl / self.config.initial_capital,
                        self.trades_today / max(1, self.config.max_trades_per_day),
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        return obs.astype(np.float32)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._reset_episode_state()
        day_index = None
        if options:
            day_index = options.get("day_index")
        if day_index is None and self.forced_day_index is not None:
            day_index = self.forced_day_index
            self.forced_day_index = None
        day_idx = self._select_day(day_index)
        self.day_start = int(day_idx[0])
        self.day_end = int(day_idx[-1])
        self.ptr = self.day_start
        self.prev_price = float(self.prices[self.ptr])
        obs = self._get_obs(self.ptr)
        return obs, {}

    def set_day_index(self, day_index: int) -> None:
        self.forced_day_index = day_index

    def step(self, action: int):
        price = float(self.prices[self.ptr])
        reward = 0.0
        size_oz = self.position_size_oz

        if self.prev_price is not None:
            pnl = (price - self.prev_price) * self.position * size_oz
            self.equity += pnl
            self.daily_pnl = self.equity - self.start_equity
            self.peak_equity = max(self.peak_equity, self.equity)
            reward += pnl

        target_pos = 0
        if action == 1:
            target_pos = 1
        elif action == 2:
            target_pos = -1

        if target_pos != self.position:
            new_size = self._position_size()
            if target_pos == 0:
                cost = abs(self.position) * self.position_size_oz
                cost *= self.config.transaction_cost_per_oz
                self.position_size_oz = 0.0
            elif self.position == 0:
                cost = abs(target_pos) * new_size
                cost *= self.config.transaction_cost_per_oz
                self.position_size_oz = new_size
            else:
                close_cost = abs(self.position) * self.position_size_oz
                open_cost = abs(target_pos) * new_size
                cost = (close_cost + open_cost) * self.config.transaction_cost_per_oz
                self.position_size_oz = new_size
            self.equity -= cost
            self.daily_pnl -= cost
            self.trades_today += 1
            self.position = target_pos
            self.entry_price = price if self.position != 0 else np.nan
            reward -= cost

        drawdown = 0.0
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - self.equity) / self.peak_equity
        reward -= self.config.drawdown_penalty * drawdown

        if self.trades_today > self.config.soft_trade_limit:
            reward -= self.config.flip_penalty * (
                self.trades_today - self.config.soft_trade_limit
            )

        terminated = False
        truncated = False

        if self.daily_pnl <= -self.config.daily_loss_limit:
            reward -= self.config.loss_limit_penalty
            terminated = True

        if self.trades_today > self.config.max_trades_per_day:
            reward -= self.config.overtrade_penalty
            terminated = True

        self.prev_price = price
        self.ptr += 1

        if self.ptr > self.day_end:
            if self.position != 0:
                close_cost = (
                    abs(self.position)
                    * self.position_size_oz
                    * self.config.transaction_cost_per_oz
                )
                self.equity -= close_cost
                self.daily_pnl -= close_cost
                reward -= close_cost
                self.position = 0
                self.position_size_oz = 0.0
                self.entry_price = np.nan
            if (
                self.daily_pnl >= self.config.profit_target
                and drawdown <= self.config.drawdown_target
            ):
                reward += self.config.end_day_bonus * (
                    self.daily_pnl / self.config.profit_target
                )
            truncated = True

        obs_idx = min(self.ptr, self.day_end)
        obs = self._get_obs(obs_idx)
        info = {
            "daily_pnl": float(self.daily_pnl),
            "trades": int(self.trades_today),
            "equity": float(self.equity),
            "drawdown": float(drawdown),
        }
        return obs, float(reward), terminated, truncated, info

from typing import List

import numpy as np
import pandas as pd
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_gold_trading.config import EnvConfig
from rl_gold_trading.envs import XAUUSDTradingEnv


def make_base_env(
    df: pd.DataFrame, feature_cols: List[str], config: EnvConfig, random_reset: bool
) -> DummyVecEnv:
    def _env_fn():
        return Monitor(XAUUSDTradingEnv(df, feature_cols, config, random_reset))

    return DummyVecEnv([_env_fn])


def make_train_env(
    df: pd.DataFrame, feature_cols: List[str], config: EnvConfig
) -> VecNormalize:
    vec_env = make_base_env(df, feature_cols, config, random_reset=True)
    return VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)


def _unwrap_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def get_unwrapped_env(vec_env: VecNormalize):
    base_env = vec_env.venv if isinstance(vec_env, VecNormalize) else vec_env
    return _unwrap_env(base_env.envs[0])


def get_day_count(vec_env: VecNormalize) -> int:
    return len(get_unwrapped_env(vec_env).day_indices)


def vec_reset(vec_env, options=None):
    if options is None:
        result = vec_env.reset()
    else:
        result = vec_env.reset(options=options)
    if isinstance(result, tuple):
        return result
    return result, {}


def vec_step(vec_env, action):
    result = vec_env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = np.logical_or(terminated, truncated)
        return obs, reward, done, info
    obs, reward, done, info = result
    return obs, reward, done, info

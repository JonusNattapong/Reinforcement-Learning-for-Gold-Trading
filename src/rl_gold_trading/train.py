from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from rl_gold_trading.config import TrainConfig


def build_model(train_env: VecNormalize, cfg: TrainConfig) -> PPO:
    return PPO(
        "MlpPolicy",
        train_env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        verbose=1,
    )


def train_model(model: PPO, cfg: TrainConfig) -> PPO:
    model.learn(total_timesteps=cfg.timesteps)
    return model

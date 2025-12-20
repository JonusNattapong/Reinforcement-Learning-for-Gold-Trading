import argparse
import os
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from rl_gold_trading.config import DataConfig, EnvConfig, TrainConfig
from rl_gold_trading.data import load_data, split_by_year
from rl_gold_trading.features import add_features
from rl_gold_trading.metrics import evaluate_model
from rl_gold_trading.train import build_model, train_model
from rl_gold_trading.vec_env import get_day_count, make_base_env, make_train_env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate PPO on XAUUSD M15.")
    parser.add_argument(
        "--csv",
        default=os.environ.get("XAUUSD_CSV"),
        help="Optional CSV path for data; falls back to Hugging Face dataset.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=int(os.environ.get("TOTAL_TIMESTEPS", "1000000")),
        help="Total PPO timesteps to train.",
    )
    parser.add_argument(
        "--save-dir",
        default=os.environ.get("MODEL_DIR", "models"),
        help="Directory to save model and VecNormalize stats.",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "train_eval"],
        default="train_eval",
        help="Run mode.",
    )
    return parser.parse_args()


def _load_vecnormalize(path: str, env) -> VecNormalize:
    vec_env = VecNormalize.load(path, env)
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env


def _load_model(path: str) -> PPO:
    return PPO.load(path)


def main(args: Optional[argparse.Namespace] = None) -> None:
    args = args or _parse_args()
    data_cfg = DataConfig(csv_path=args.csv)
    env_cfg = EnvConfig()
    train_cfg = TrainConfig(timesteps=args.timesteps, save_dir=args.save_dir)

    raw_df = load_data(data_cfg)
    feat_df, feature_cols = add_features(raw_df)
    train_df, valid_df, test_df = split_by_year(feat_df, data_cfg)

    train_env = make_train_env(train_df, feature_cols, env_cfg)
    eval_env_valid = make_base_env(valid_df, feature_cols, env_cfg, random_reset=False)
    eval_env_test = make_base_env(test_df, feature_cols, env_cfg, random_reset=False)

    model_path = os.path.join(train_cfg.save_dir, "ppo_xauusd")
    vec_path = os.path.join(train_cfg.save_dir, "vecnormalize.pkl")

    if args.mode in ("train", "train_eval"):
        model = build_model(train_env, train_cfg)
        model = train_model(model, train_cfg)
        os.makedirs(train_cfg.save_dir, exist_ok=True)
        model.save(model_path)
        train_env.save(vec_path)
    else:
        model = _load_model(model_path)

    eval_env_valid = _load_vecnormalize(vec_path, eval_env_valid)
    eval_env_test = _load_vecnormalize(vec_path, eval_env_test)

    valid_stats = evaluate_model(
        model, eval_env_valid, get_day_count(eval_env_valid), env_cfg.initial_capital
    )
    test_stats = evaluate_model(
        model, eval_env_test, get_day_count(eval_env_test), env_cfg.initial_capital
    )

    print("VALID:", valid_stats)
    print("TEST:", test_stats)


if __name__ == "__main__":
    main()

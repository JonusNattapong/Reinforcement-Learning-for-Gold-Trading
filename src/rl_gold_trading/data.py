import os
from typing import Optional, Tuple

import pandas as pd

from rl_gold_trading.config import DataConfig


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    date_col = None
    for key in ["datetime", "date", "time", "timestamp"]:
        if key in cols:
            date_col = cols[key]
            break
    if date_col is None:
        raise ValueError("No datetime column found in dataset.")
    required = ["open", "high", "low", "close"]
    missing = [col for col in required if col not in cols]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.rename(
        columns={
            cols.get("open", "open"): "open",
            cols.get("high", "high"): "high",
            cols.get("low", "low"): "low",
            cols.get("close", "close"): "close",
            cols.get("volume", "volume"): "volume",
            date_col: "datetime",
        }
    )
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
    df = df.sort_values("datetime").drop_duplicates("datetime")
    df = df.set_index("datetime")
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    df = df.sort_index()
    ohlc = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    df = df.resample(rule).agg(ohlc)
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


def load_xauusd_from_hf() -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets is required for Hugging Face loading.") from exc
    dataset = load_dataset("ZombitX64/xauusd-gold-price-historical-data-2004-2025")
    if "train" in dataset:
        df = dataset["train"].to_pandas()
    else:
        split = list(dataset.keys())[0]
        df = dataset[split].to_pandas()
    return df


def load_data(cfg: DataConfig) -> pd.DataFrame:
    if cfg.csv_path and os.path.isfile(cfg.csv_path):
        df = pd.read_csv(cfg.csv_path)
    else:
        df = load_xauusd_from_hf()
    df = _standardize_columns(df)
    df = _resample(df, cfg.resample_rule)
    return df


def split_by_year(df: pd.DataFrame, cfg: DataConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = df.index.year
    train = df[(years >= cfg.train_start_year) & (years <= cfg.train_end_year)]
    valid = df[(years >= cfg.valid_start_year) & (years <= cfg.valid_end_year)]
    test = df[(years >= cfg.test_start_year) & (years <= cfg.test_end_year)]
    return train, valid, test

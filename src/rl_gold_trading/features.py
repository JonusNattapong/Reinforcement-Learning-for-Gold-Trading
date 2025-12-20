from typing import List, Tuple

import numpy as np
import pandas as pd


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss.replace(0.0, np.nan))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def add_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df["log_return"] = np.log(df["close"]).diff()
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["body"] = (df["close"] - df["open"]) / df["close"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr14"] = tr.rolling(14).mean() / df["close"]
    df["rsi14"] = _rsi(df["close"], 14) / 100.0
    ema20 = df["close"].ewm(span=20, adjust=False).mean()
    ema50 = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_diff"] = (ema20 - ema50) / df["close"]
    df["volatility"] = df["log_return"].rolling(32).std()
    minutes = df.index.hour * 60 + df.index.minute
    day_frac = minutes / (24 * 60)
    df["tod_sin"] = np.sin(2 * np.pi * day_frac)
    df["tod_cos"] = np.cos(2 * np.pi * day_frac)
    feature_cols = [
        "log_return",
        "hl_range",
        "body",
        "atr14",
        "rsi14",
        "ema_diff",
        "volatility",
        "tod_sin",
        "tod_cos",
    ]
    df = df.dropna(subset=feature_cols)
    return df, feature_cols

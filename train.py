import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parent
    src_path = root / "src"
    sys.path.insert(0, str(src_path))


def main() -> None:
    _ensure_src_on_path()
    from rl_gold_trading.run import main as run_main

    run_main()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run DynaFE-Net on both dynamic gas mixture tasks.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--preset", default="quick", choices=["quick", "full"])
    parser.add_argument("--extra", nargs=argparse.REMAINDER, help="Extra args passed to train.py")
    args = parser.parse_args()

    mixtures = ["ethylene_co", "ethylene_methane"]
    root = Path(__file__).resolve().parent
    train_py = root / "train.py"

    for mixture in mixtures:
        cmd = [
            sys.executable,
            str(train_py),
            "--data-dir",
            args.data_dir,
            "--results-dir",
            args.results_dir,
            "--preset",
            args.preset,
            "--mixture",
            mixture,
        ]

        if args.extra:
            extra = args.extra
            if extra and extra[0] == "--":
                extra = extra[1:]
            cmd.extend(extra)

        print("\n[run_all] Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import json
from pathlib import Path

from main_code import (
    run_smooth_experiments,
    run_subgradient_experiments,
    sanity_check,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the non-bonus ORIE 6365 practical assignment experiments.")
    parser.add_argument("--out_root", type=str, default="outputs")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--m", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=6365)
    parser.add_argument("--smooth_iters", type=int, default=250)
    parser.add_argument("--subgrad_iters", type=int, default=400)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    sanity = sanity_check(seed=args.seed)
    with open(out_root / "sanity_check.json", "w", encoding="utf-8") as f:
        json.dump(sanity, f, indent=2)

    smooth = run_smooth_experiments(
        n=args.n,
        m=args.m,
        sigma_values=(10.0, 1e3, 1e5),
        mu_values=(0.0, 1e-3, 1e-1),
        n_iters=args.smooth_iters,
        seed=args.seed,
        out_dir=str(out_root / "smooth"),
        use_scipy_reference=True,
    )
    with open(out_root / "smooth_summary_copy.json", "w", encoding="utf-8") as f:
        json.dump(smooth, f, indent=2)

    subgrad = run_subgradient_experiments(
        n=args.n,
        m=args.m,
        sigma_values=(10.0, 1e3, 1e5),
        R_values=(1.0, 5.0),
        gamma_values=(0.01, 0.05),
        n_iters=args.subgrad_iters,
        seed=args.seed,
        out_dir=str(out_root / "subgradient"),
    )
    with open(out_root / "subgradient_summary_copy.json", "w", encoding="utf-8") as f:
        json.dump(subgrad, f, indent=2)

    print("Finished. Results stored in:", out_root)

if __name__ == "__main__":
    main()

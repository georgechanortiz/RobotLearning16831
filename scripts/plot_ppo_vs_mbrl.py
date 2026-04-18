# Example usage:
# python scripts/plot_ppo_vs_mbrl.py

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PPO_CSV = Path("logs/go2_flat_ppo_2026-03-18_11-18-16_ppo_torch.csv")
MBRL_CSV = Path("logs/mbrl/go2_walk_2026-04-14_20-34-50/metrics.csv")
OUTPUT_PNG = Path("ppo_vs_mbrl_return.png")


def main() -> None:
    ppo = pd.read_csv(PPO_CSV)
    mbrl = pd.read_csv(MBRL_CSV)

    plt.figure(figsize=(9, 6))
    plt.plot(ppo["Step"], ppo["Value"], label="PPO Return", linewidth=2)
    plt.plot(mbrl["env_steps"], mbrl["mean_return_100"], label="MBRL Mean Return (100 ep)", linewidth=2)
    plt.plot(
        mbrl["env_steps"],
        mbrl["best_mean_return"],
        label="MBRL Best Mean Return",
        linestyle="--",
        linewidth=1.8,
    )

    plt.title("Go2 Walking: PPO vs MBRL")
    plt.xlabel("Training Steps")
    plt.ylabel("Return")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=200)
    plt.show()


if __name__ == "__main__":
    main()

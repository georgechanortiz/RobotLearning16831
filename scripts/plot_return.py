# Example Usage
# python scripts/plot_return.py

import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

rand_agent_filepath = "logs/go2_flat_ppo_2026-02-08_14-15-29_ppo_torch_rand.csv"
ppo_agent_filepath = "logs/go2_flat_ppo_2026-03-18_11-18-16_ppo_torch.csv"
sac_agent_filepath = "logs/skrl_go2_flat_sac_2026-03-22_21-23-29_sac_torch-total_reward_mean.csv"

rand_agent = pd.read_csv(rand_agent_filepath)
ppo_agent = pd.read_csv(ppo_agent_filepath)
sac_agent = pd.read_csv(sac_agent_filepath)


rand_mean = rand_agent["Value"].mean()
# Average Return across Iterations
plt.figure(figsize=(8,6))
plt.axhline(y=rand_mean, color='r', linestyle='--', label="Random Agent (Baseline)")
plt.plot(ppo_agent["Step"], ppo_agent["Value"], label="PPO Agent")
plt.plot(sac_agent["Step"], sac_agent["Value"], label="SAC Agent")

plt.title("Average Return across Iterations")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Average Return")
plt.grid(True)
plt.tight_layout()

plt.savefig("average_return_across_iterations.png")
plt.show()
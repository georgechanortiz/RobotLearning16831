# Example Usage
# python scripts/plot_return.py

import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

path = "logs/skrl/go2_flat_ppo/2026-02-08_14-15-29_ppo_torch/events.out.tfevents.1770578133.rml.3925001.0"

ea = event_accumulator.EventAccumulator(path)
ea.Reload()

tag = "Reward / Total reward (mean)"
# print(ea.Tags())
times, steps, values = zip(*[(e.wall_time, e.step, e.value) for e in ea.Scalars(tag)])

plt.figure()
plt.plot(steps, values)
plt.xlabel("Environment Step")
plt.ylabel("Average Return")
plt.title("Agent Performance")
plt.savefig('agent_performance.png')
plt.show()

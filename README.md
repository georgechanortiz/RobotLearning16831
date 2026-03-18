# FP16831 — Robot Learning with Isaac Lab

## Overview

This project builds on Isaac Lab to train RL policies for Unitree robots in simulation.
It supports both **quadruped** (Go2) and **humanoid** (H1, G1) robots, with training via
[RSL-RL](https://github.com/leggedrobotics/rsl_rl) and [skrl](https://skrl.readthedocs.io).

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, isaaclab

---

## Available Robots

Robot configurations are defined in `source/FP16831/FP16831/robots/unitree.py`.

### Quadrupeds

| Config | Description |
|---|---|
| `UNITREE_A1_CFG` | Unitree A1 with DC motor model |
| `UNITREE_GO1_CFG` | Unitree Go1 with MLP actuator net |
| `UNITREE_GO2_CFG` | Unitree Go2 with DC motor model |

### Humanoids

| Config | Description |
|---|---|
| `H1_CFG` | Unitree H1 humanoid (full collision meshes) |
| `H1_MINIMAL_CFG` | H1 with minimal collision meshes (faster simulation) |
| `G1_CFG` | Unitree G1 humanoid (full collision meshes) |
| `G1_MINIMAL_CFG` | G1 with minimal collision meshes (faster simulation) |
| `G1_29DOF_CFG` | G1 29-DOF for locomanipulation (configurable fixed/mobile base) |
| `G1_INSPIRE_FTP_CFG` | G1 29-DOF with Inspire 5-finger hand (fixed base, for grasping) |

**H1** actuator groups: legs (hip yaw/roll/pitch, knee, torso), feet (ankle), arms (shoulder, elbow).

**G1** actuator groups: legs, feet, arms (+ finger joints). The `G1_29DOF_CFG` variant adds waist and hand
actuator groups and uses `DCMotorCfg` for legs/feet with per-joint effort/velocity limits tuned for
locomanipulation. Toggle `fix_root_link` for fixed-base (upper-body only) vs. mobile (locomotion +
manipulation) scenarios:

```python
from FP16831.robots.unitree import G1_29DOF_CFG

# Fixed base (upper-body manipulation only)
fixed_cfg = G1_29DOF_CFG.copy()
fixed_cfg.spawn.articulation_props.fix_root_link = True

# Mobile (locomotion + manipulation)
mobile_cfg = G1_29DOF_CFG.copy()
mobile_cfg.spawn.articulation_props.fix_root_link = False
```

---

## Registered Gym Environments

Defined in `source/FP16831/FP16831/tasks/manager_based/fp16831/__init__.py`:

| Task ID | Robot | Description | RL Config |
|---|---|---|---|
| `Template-Fp16831-v0` | Cartpole | Default template task | RSL-RL PPO, skrl PPO, skrl AMP |
| `Random-Agent-Unitree-Go2-v0` | Go2 | Flat terrain velocity tracking | skrl PPO |
| `Random-Agent-Unitree-Go2-Play-v0` | Go2 | Flat terrain (evaluation, 50 envs) | skrl PPO |

> **Note:** `scripts/list_envs.py` only shows tasks whose ID contains `"Template-"`. The Go2 tasks
> use the `"Random-Agent-"` prefix and will not appear. To list them, update the filter in
> `list_envs.py` line 55, or pass `--keyword` with a different prefix.

---

## Running Humanoid Experiments

The humanoid robot configurations are ready to use but **no humanoid-specific Gym task is currently registered**.
To run experiments with the H1 or G1 humanoids, create a new environment config that swaps the robot,
following the same pattern used for Go2.

### Step 1: Create a Humanoid Environment Config

Add a new file (e.g. `h1_flat_env_cfg.py`) alongside the existing configs in
`source/FP16831/FP16831/tasks/manager_based/fp16831/`:

```python
from isaaclab.utils import configclass
from FP16831.robots.unitree import H1_MINIMAL_CFG  # or G1_CFG, G1_29DOF_CFG, etc.
from FP16831.config.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

@configclass
class UnitreeH1FlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Swap robot
        self.scene.robot = H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/pelvis"
        # Flat terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None
        # Adjust rewards for humanoid contact bodies
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_ankle_link"
        self.terminations.base_contact.params["sensor_cfg"].body_names = "pelvis"
```

For the **G1**, import `G1_CFG`, `G1_MINIMAL_CFG`, or `G1_29DOF_CFG` instead and adjust body names
(`pelvis`, `.*_ankle_roll_link`, etc.) to match the G1 URDF.

### Step 2: Register the Gym Environment

In `source/FP16831/FP16831/tasks/manager_based/fp16831/__init__.py`, add:

```python
gym.register(
    id="Template-Unitree-H1-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_flat_env_cfg:UnitreeH1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
```

### Step 3: Train

Use either RL framework:

```bash
# RSL-RL (PPO)
python scripts/rsl_rl/train.py --task=Template-Unitree-H1-Flat-v0 --num_envs=4096 --headless

# skrl (PPO)
python scripts/skrl/train.py --task=Template-Unitree-H1-Flat-v0 --num_envs=4096 --headless

# skrl (AMP — suited for motion-imitation on humanoids)
python scripts/skrl/train.py --task=Template-Unitree-H1-Flat-v0 --algorithm=AMP --num_envs=4096 --headless
```

An AMP agent config (`skrl_amp_cfg.yaml`) with larger networks (1024 × 512) and a discriminator is
already provided in `agents/` and logs to `humanoid_amp_run`.

### Step 4: Evaluate / Play

```bash
# RSL-RL
python scripts/rsl_rl/play.py --task=Template-Unitree-H1-Flat-v0 --num_envs=16

# skrl (specify checkpoint)
python scripts/skrl/play.py --task=Template-Unitree-H1-Flat-v0 --num_envs=16 \
    --checkpoint=logs/skrl/humanoid_amp_run/<run_dir>/checkpoints/best_agent.pt
```

### Key CLI Options

| Flag | Description |
|---|---|
| `--task` | Gym environment ID |
| `--num_envs` | Number of parallel environments |
| `--headless` | Run without GUI |
| `--video` | Record video |
| `--max_iterations` | Training iterations (RSL-RL) |
| `--seed` | RNG seed |
| `--distributed` | Multi-GPU training |
| `--algorithm` | skrl algorithm (`PPO`, `AMP`, `IPPO`, `MAPPO`) |
| `--ml_framework` | skrl backend (`torch`, `jax`) |

---

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/FP16831

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/FP16831/FP16831/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/FP16831"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```
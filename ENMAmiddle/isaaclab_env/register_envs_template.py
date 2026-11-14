"""Gymnasium registration for the ENMA environments -- TEMPLATE.

The original registration lived inside the lab's copy of `unitree_rl_lab` and was not part of the code package,
so this file was reconstructed from how the scripts use the task ids:

    train_enma.py / run_evaluation.py :  "Isaac-ENMA-Go2-v0"
    play.py --video / --cameratest    :  "Isaac-ENMA-Go2-Moving-Interaction-v0"

TODO (verify before use): the mapping from task id to configuration class below is an assumption.

How to use
----------
1. Copy `enma_env_cfg.py` to
   `unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/enma_env_cfg.py`.
2. Add the two `gym.register(...)` calls below to the `__init__.py` of that `go2` package
   (the scripts trigger registration with `import unitree_rl_lab.tasks`).
3. Set the environment variable ENMA_ROOT to this repository's root so the environment finds
   `checkpoints/low_level_locomotion/policy.pt` and the human avatar asset.
"""

import gymnasium as gym

_ENV_MODULE = "unitree_rl_lab.tasks.locomotion.robots.go2.enma_env_cfg"

gym.register(
    id="Isaac-ENMA-Go2-v0",
    entry_point=f"{_ENV_MODULE}:ENMAEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{_ENV_MODULE}:ENMAEnvCfg"},  # assumption, see TODO above
)

gym.register(
    id="Isaac-ENMA-Go2-Moving-Interaction-v0",
    entry_point=f"{_ENV_MODULE}:ENMAEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{_ENV_MODULE}:ENMATestCfg"},  # assumption, see TODO above
)

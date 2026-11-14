# Architecture

This note explains how the pieces of the repository fit together. Everything here describes the code as written.

## 1. Three layers of control

```
high-level:  navigation planner or human teleoperator      -> nominal command  u  (vx, vy, yaw)
mid-level :  ENMA (this work)                               -> bounded correction a (6 values)
low-level :  pre-trained walking policy / factory controller -> 12 joint targets
```

ENMA never outputs joint commands. It only shifts the command that the walking controller receives, so the
walking controller can be swapped without retraining ENMA. In simulation the walking controller is the exported
policy in `checkpoints/low_level_locomotion/`; on the real Go2 it is Unitree's factory controller.

## 2. What ENMA sees and outputs

**Input (12 values):** nominal command (3), robot base velocity (3), human position relative to the robot (2),
human velocity relative to the robot (2), a task-state flag (1) and the distance to the human (1).

**Output (6 values, each bounded by `tanh`):**

| Channel | Bound | Used as |
|---|---|---|
| 1 | +-0.3 | correction to forward velocity |
| 2 | +-0.2 | correction to lateral velocity |
| 3 | +-0.1 | correction added to the third command channel |
| 4 | +-0.3 | body roll correction |
| 5 | +-0.3 | body pitch correction (scaled by 0.3 in the simulator) |
| 6 | +-0.5 | body yaw correction (scaled by 0.3 in the simulator) |

**Fusion:** `command = nominal + correction` for the first three channels (correction only when the robot has no
navigation goal); the last three channels are passed on after scaling. See `_step_impl` in
`ENMAmiddle/isaaclab_env/enma_env_cfg.py`.

## 3. The spiking policy

The actor is a population-coded spiking network in the style of POPSAN: four fully connected layers of 256 leaky
integrate-and-fire neurons, simulated for 16 timesteps per control step. The spikes of the last layer are averaged
into a firing-rate vector, a linear readout maps it to six real numbers, and `tanh` bounds them. Gradients pass
through the spikes with a fast-sigmoid surrogate. The critic is an ordinary MLP. PPO treats the actor output as the
mean of a Gaussian with a learned standard deviation.

## 4. The training environment

`ENMAEnv` (Isaac Lab, 64 parallel copies, 20 s episodes) contains a Unitree Go2 and a humanoid avatar that carries
a first-person camera (256x256). At every reset one of five human scenarios is drawn at random:

| Scenario | Human behaviour |
|---|---|
| approach | walks toward the robot and stops about 1 m away |
| stand | stands close to the robot |
| orbit | circles the robot |
| cross | walks past the robot from one side to the other |
| navigation | robot and human both walk between waypoints |

In the navigation scenario the robot follows a nominal command toward its goal; in the others it has no goal and
ENMA's correction is the whole command. In the navigation scenario the human is driven by a minimal Social
Force Model (Helbing and Molnar) with attraction to its goal and repulsion from the robot and the walls.

## 5. Where the reward comes from

```
reward per step = locomotion terms (Isaac Lab reward manager)
                + avoidance term (move when the human is close, penalty within 0.3 m)
                + affect bonus (same value for all steps of a subwindow)

affect bonus    = 0.375 x [ (V + A) + (dV + dA) ]      V, A = Caesar's valence and arousal for the subwindow
```

The camera is sampled every 4 simulation steps, converted to grayscale 180x320, grouped into 16-frame clips and
scored by Caesar; a rollout is divided into three subwindows. `dV`, `dA` are the changes from the previous
subwindow. Caesar itself was trained offline to imitate the TRIBE v2 + SwiFT teacher, so the reward needs no human
in the loop.

## 6. Deployment

`ENMAmiddle/deploy/enma_inference_node.py` is a ROS 2 node for the Go2. It reads the wireless controller, runs
the actor at 20 Hz using the policy mean, clamps the output and publishes a `geometry_msgs/Twist` on `/cmd_vel`.
L2 acts as a dead-man switch and R2 as an emergency stop.

## 7. Evaluation

`evaluation/run_evaluation.py` runs the Nominal and A* planners with and without ENMA under identical seeds and
writes per-episode metrics. See `evaluation/README.md`.

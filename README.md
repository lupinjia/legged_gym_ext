# legged_gym extension #
This repository is an extension of original [legged_gym](https://github.com/leggedrobotics/legged_gym), containing my custom implementation of published papers.

---

## Implemented papers

| Paper Title | Authors | Year | Folder |
|-------------|---------|------|--------|
| [Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion](https://arxiv.org/abs/2202.05481) | Ji et. al. | 2022 | [go2_ee](https://github.com/lupinjia/legged_gym_ext/tree/master/legged_gym/envs/go2/go2_ee) |
| [Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition](https://arxiv.org/abs/2011.01387) | Siekmann et. al. | 2021 | [go2_multi_gait](https://github.com/lupinjia/legged_gym_ext/tree/master/legged_gym/envs/go2/go2_multi_gait) |

## Installation ###
1. Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended)
2. Install [pytorch with cuda](https://pytorch.org/get-started/locally/)
3. Install Isaac Gym
   - Download and install Isaac Gym from https://developer.nvidia.com/isaac-gym
4. Install rsl_rl_ext
   - Clone https://github.com/lupinjia/rsl_rl_ext
   -  `cd rsl_rl_ext && pip install -e .` 
5. Install legged_gym
    - Clone this repository
   - `pip install -e .` (under legged_gym_ext)


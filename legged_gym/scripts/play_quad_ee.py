import sys
sys.path.append("/home/jason/unitree_rl_gym/legged_gym")
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger, export_estimator_as_jit

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.border_size = 10.0
    env_cfg.terrain.static_friction = 1.0
    env_cfg.terrain.dynamic_friction = 1.0
    env_cfg.terrain.restitution = 0.0
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.selected = True
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.push_robots = False
    # env_cfg.domain_rand.push_interval_s = 1
    env_cfg.domain_rand.push_rigid_bodies = False
    # env_cfg.domain_rand.push_rb_interval_s = 1
    env_cfg.domain_rand.randomize_com_pos = False
    env_cfg.asset.fix_base_link = False
    env_cfg.init_state.init_state_train = False
    env_cfg.init_state.init_state_play = True

    env_cfg.env.test = True
    env_cfg.env.episode_length_s = 20
    env_cfg.viewer.debug_viz = True
    #----- prepare environment -----#
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    estimator_input, critic_obs = env.get_observations()
    
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    estimator = ppo_runner.get_inference_estimator(device=env.device)
    
    #----- Logger config -----#
    logger = Logger(env.dt)
    robot_index = 0
    joint_index = 2  # FL_calf
    stop_state_log = 400
    
    #---------- FOLLOW_ROBOT Camera Config ----------#
    camera_lookat_follow = np.array(env_cfg.viewer.lookat)
    camera_deviation_follow = np.array([1., 0., 0.])
    camera_position_follow = camera_lookat_follow - camera_deviation_follow
    
    #---------- Change spd ----------#
    change_spd_interval = 200          # time interval to change speed [s]
    env.command_ranges["lin_vel_x"] = [0.3, 0.3]
    env.command_ranges["lin_vel_y"][0] = env.command_ranges["lin_vel_y"][1] = 0.0
    env.command_ranges["ang_vel_yaw"][0] = env.command_ranges["ang_vel_yaw"][1] = 0.0
    env.command_ranges["heading"][0] = env.command_ranges["heading"][1] = 0.0
    env._resample_commands(np.arange(env.num_envs))
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'estimator')
        export_estimator_as_jit(estimator, path)
        print("Exported estimator network as jit script to: ", path)
    
    #---------- Agent Env Interaction Loop ----------#
    for i in range(10*int(env.max_episode_length)):
        estimator_output = estimator(estimator_input)
        actor_obs = torch.cat((estimator_input, estimator_output), dim=-1)
        policy_actions = policy(actor_obs.detach())
        # actions = torch.zeros_like(policy_actions)
        estimator_input, estimator_true_value, critic_obs, _, rews, dones, infos = env.step(policy_actions.detach())
        
        if FOLLOW_ROBOT:
            # refresh where camera looks at(robot 0 base)
            camera_lookat_follow = env.root_states[robot_index, 0:3].cpu().numpy()
            # refresh camera's position
            camera_position_follow = camera_lookat_follow - camera_deviation_follow
            env.set_camera(camera_position_follow, camera_lookat_follow)
        
        # print info
        # print(f"base height: {env.root_states[robot_index, 2].item():.3f}")
        print(f"feet pos: {env.foot_pos[robot_index, :, 2]}")
        # print(f"foot vel: {env.foot_vel[robot_index, :]}")
        # print(f"pitch angle: {env.rpy[robot_index, 1]:.3f}")
        # print(f"phi: {env.phi[robot_index, 0].item():.3f}")
        # print(f"gait_time: {env.gait_time[robot_index, 0].item():.3f}")
        # print(f"gait_period: {env.gait_period[robot_index, 0].item():.3f}")
        # print(f"history_obs: {env.history_obs[:, robot_index, :]}")
        # print(f"history_obs_buf: {env.history_obs_buf[robot_index, :]}")
            
        #---------- logging ----------#
        if i < stop_state_log:
            logger.log_states({
                
                ## log to plot ##
                # GRF and C_frc, C_spd
                "GRF_fl": env.foot_contact_force_fl[0, 2].item(),
                "GRF_fr": env.foot_contact_force_fr[0, 2].item(),
                "GRF_rl": env.foot_contact_force_rl[0, 2].item(),
                "GRF_rr": env.foot_contact_force_rr[0, 2].item(),
                "E[C_frc_fl]": env.exp_C_frc_fl[0].item(),
                "E[C_frc_fr]": env.exp_C_frc_fr[0].item(),
                "E[C_frc_rl]": env.exp_C_frc_rl[0].item(),
                "E[C_frc_rr]": env.exp_C_frc_rr[0].item(),
                
                "vel_cmd_x": env.commands[robot_index, 0].item(),
                "vel_cmd_y": env.commands[robot_index, 1].item(),
                "vel_cmd_yaw": env.commands[robot_index, 2].item(),
                # "base_roll": env.rpy[robot_index, 0].item(),
                # "base_pitch": env.rpy[robot_index, 1].item(),
                # "base_yaw": env.rpy[robot_index, 2].item(),
                # "base_lin_vel_x": env.base_lin_vel[robot_index, 0].item(),
                # "base_lin_vel_y": env.base_lin_vel[robot_index, 1].item(),
                # "base_lin_vel_z": env.base_lin_vel[robot_index, 2].item(),
                
                
                # joint desired pos, actual pos, default pos
                "desired_left_hip_dof_pos": env.actions[robot_index, 0].item()*env_cfg.control.action_scale+env.default_dof_pos[robot_index, 0].item(),
                "desired_left_thigh_dof_pos": env.actions[robot_index, 1].item()*env_cfg.control.action_scale+env.default_dof_pos[robot_index, 1].item(),
                "desired_left_calf_dof_pos": env.actions[robot_index, 2].item()*env_cfg.control.action_scale+env.default_dof_pos[robot_index, 2].item(),
                "desired_right_hip_dof_pos": env.actions[robot_index, 3].item()*env_cfg.control.action_scale+env.default_dof_pos[robot_index, 3].item(),
                "desired_right_thigh_dof_pos": env.actions[robot_index, 4].item()*env_cfg.control.action_scale+env.default_dof_pos[robot_index, 4].item(),
                "desired_right_calf_dof_pos": env.actions[robot_index, 5].item()*env_cfg.control.action_scale+env.default_dof_pos[robot_index, 5].item(),
                
                # joint torques
                "fl_hip_torque": env.torques[robot_index, 0].item(),
                "fl_thigh_torque": env.torques[robot_index, 1].item(),
                "fl_calf_torque": env.torques[robot_index, 2].item(),
                "fr_hip_torque": env.torques[robot_index, 3].item(),
                "fr_thigh_torque": env.torques[robot_index, 4].item(),
                "fr_calf_torque": env.torques[robot_index, 5].item(),
                "rl_hip_torque": env.torques[robot_index, 6].item(),
                "rl_thigh_torque": env.torques[robot_index, 7].item(),
                "rl_calf_torque": env.torques[robot_index, 8].item(),
                "rr_hip_torque": env.torques[robot_index, 9].item(),
                "rr_thigh_torque": env.torques[robot_index, 10].item(),
                "rr_calf_torque": env.torques[robot_index, 11].item(),
                
            })
        elif i == stop_state_log:
            # pass
            logger.plot_states()
            # logger.save_data_to_xlsx()
        

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    FOLLOW_ROBOT = False
    args = get_args()
    play(args)
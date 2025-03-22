from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
import numpy as np
import os
import sys

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math import wrap_to_pi, quat_apply_yaw
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.helpers import class_to_dict
from scipy.stats import vonmises

class GO2EE(LeggedRobot):
    
    def get_observations(self):
        return self.estimator_input_buf, self.critic_obs_buf
    
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.estimator_input_buf = torch.clip(self.estimator_input_buf, -clip_obs, clip_obs);
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.estimator_input_buf, self.estimator_true_value, self.critic_obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        estimator_input, estimator_true_value, critic_obs_buf, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return estimator_input, estimator_true_value, critic_obs_buf
    
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)  # Periodic Reward Framework
        # the wrapped tensor will be updated automatically once you call refresh_xxx_tensor

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self._refresh_rigid_body_states()
        for i in range(len(self.feet_names)):
            self.p_b2f[:, 3*i:3*(i+1)] = quat_rotate_inverse(self.base_quat, self.foot_pos[:, i] - self.base_pos) # foot pos relative to base pos(base frame)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        # Periodic Reward Framework phi cycle
        # step after computing reward but before resetting the env
        self.gait_time += self.dt
        is_over_limit = (self.gait_time > self.gait_period + (self.dt / 2))  # +self.dt/2 in case of float precision errors
        over_limit_indices = is_over_limit.nonzero(as_tuple=False).flatten()
        self.gait_time[over_limit_indices] = self.dt
        self.phi = self.gait_time / self.gait_period
        
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.calc_periodic_reward_obs()  # Periodic Reward Framework
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        
        self._update_dof_obs_history()
        
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
    
    def _update_dof_obs_history(self):
        self.llast_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.l3ast_dof_pos[:] = self.llast_dof_pos[:]
        self.llast_dof_pos[:] = self.last_dof_pos[:]
        self.last_dof_pos[:] = self.dof_pos[:]
        self.l3ast_dof_vel[:] = self.llast_dof_vel[:]
        self.llast_dof_vel[:] = self.last_dof_vel[:]
        self.last_dof_vel[:] = self.dof_vel[:]
    
    def check_dof_pos_limit_termination(self):
        """ Check if the robot's dof positions are out of the specified range
        """
        lower_limits = torch.any(self.dof_pos < self.dof_pos_limits[:, 0], dim=1)
        upper_limits = torch.any(self.dof_pos > self.dof_pos_limits[:, 1], dim=1)
        self.reset_buf |= torch.logical_or(lower_limits, upper_limits)
        
    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        # self.check_dof_pos_limit_termination()
    
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        self.update_curriculum(env_ids)
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)
        self.resample_phase_and_theta(env_ids)

        # reset buffers
        self.last_dof_pos[env_ids] = 0.
        self.llast_dof_pos[env_ids] = 0.
        self.l3ast_dof_pos[env_ids] = 0.
        self.llast_actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.llast_dof_vel[env_ids] = 0.
        self.l3ast_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.gait_time[env_ids] = 0.  # Periodic Reward Framework.
        self.phi[env_ids] = 0  
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 达到重采样周期的env才会进行重采样
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        # env_ids = (self.episode_length_buf % int(self.cfg.rewards.periodic_reward_framework.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        # # Periodic Reward Framework. resample phase and theta
        # self.resample_phase_and_theta(env_ids)
        
        
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), *self.cfg.commands.ranges.ang_vel_yaw)
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        if self.cfg.domain_rand.push_rigid_bodies and (self.common_step_counter % self.cfg.domain_rand.push_interval_rb == 0):
            self._push_rigid_bodies()
        
    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        # self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
        self.commands[env_ids, 0] *= (torch.abs(self.commands[env_ids, 0]) > 0.1)
        self.commands[env_ids, 1] *= (torch.abs(self.commands[env_ids, 1]) > 0.1)
    
    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        if self.cfg.init_state.init_state_train:
            dof_pos = torch.zeros((len(env_ids), self.num_dof), dtype=torch.float, device=self.device) # 定义关节位置变量
            dof_pos[:, [0, 3, 6, 9]] = self.default_dof_pos[:,[0, 3, 6, 9]] + torch_rand_float(-0.2, 0.2, (len(env_ids), 4), device=self.device) # hip
            dof_pos[:, [1, 4, 7, 10]] = self.default_dof_pos[:,[1, 4, 7, 10]] + torch_rand_float(-0.5, 0.5, (len(env_ids), 4), device=self.device) # thigh
            dof_pos[:, [2, 5, 8, 11]] = self.default_dof_pos[:,[2, 5, 8, 11]] + torch_rand_float(-0.5, 0.5, (len(env_ids), 4), device=self.device) # calf
            self.dof_pos[env_ids] = dof_pos
            # self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
            self.dof_vel[env_ids] = torch_rand_float(-0.5, 0.5, (len(env_ids), self.num_dof), device=self.device)
        elif self.cfg.init_state.init_state_play:
            self.dof_pos[env_ids] = self.default_dof_pos
            self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:9] = torch_rand_float(-1.0, 1.0, (len(env_ids), 2), device=self.device) # lin xy vel
        self.root_states[env_ids, 9:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 4), device=self.device) # [9:13]: lin_z, ang vel
        # base orientation
        base_orien_scale = self.cfg.init_state.base_ang_random_scale
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(torch_rand_float(-base_orien_scale, base_orien_scale, (len(env_ids), 1), device=self.device).view(-1),
                                                             torch_rand_float(-base_orien_scale, base_orien_scale, (len(env_ids), 1), device=self.device).view(-1),
                                                             torch_rand_float(-base_orien_scale, base_orien_scale, (len(env_ids), 1), device=self.device).view(-1))
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:10] += torch_rand_float(-max_vel, max_vel, (self.num_envs, 3), device=self.device) # lin vel x/y/z
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
    
    def _push_rigid_bodies(self):
        """ Random pushes the rigid bodies. 
        """
        max_force = self.cfg.domain_rand.max_push_force
        force_tensor = max_force * (torch.rand(self.num_envs, self.num_bodies, 3, device=self.device) * 2 - 1) # force x/y/z
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(force_tensor), None, gymapi.CoordinateSpace.ENV_SPACE)
    
    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # # draw height lines
        # if not self.terrain.cfg.measure_heights:
        #     return
        # self.gym.clear_lines(self.viewer)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)
        # sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        # for i in range(self.num_envs):
        #     base_pos = (self.root_states[i, :3]).cpu().numpy()
        #     heights = self.measured_heights[i].cpu().numpy()
        #     height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
        #     for j in range(heights.shape[0]):
        #         x = height_points[j, 0] + base_pos[0]
        #         y = height_points[j, 1] + base_pos[1]
        #         z = heights[j]
        #         sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
        #         gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 
        pass
    
    def compute_observations(self):
        """ Computes observations
        """
        self.estimator_input_buf = torch.cat(( self.commands[:, :3] * self.commands_scale,                           # 3
                                   self.projected_gravity * self.obs_scales.gravity,                           # 3
                                   self.base_ang_vel  * self.obs_scales.ang_vel,                         # 3
                                  (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,       # 12
                                   self.dof_vel * self.obs_scales.dof_vel,                               # 12
                                  (self.last_dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, # 12
                                  (self.llast_dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,# 12
                                  (self.l3ast_dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,# 12
                                  self.last_dof_vel * self.obs_scales.dof_vel,                           # 12
                                  self.llast_dof_vel * self.obs_scales.dof_vel,                          # 12
                                  self.l3ast_dof_vel * self.obs_scales.dof_vel,                          # 12
                                  self.actions,                                                         # 12
                                  self.last_actions,
                                  self.clock_input,                                                     # 4
                                  self.phase_ratio,                                                     # 2
                                    ),dim=-1)
        # if torch.isnan(self.estimator_input_buf).any():
        #     print("nan in estimator_input_buf")
        #     print(f"self.phi: {self.phi}")
        #     print(f"estimator_input_buf: {self.estimator_input_buf}")
        #     sys.exit()
        self.estimator_true_value = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,              # 3
                                               self.foot_pos[:, :, 2],                                   # 4
                                               (self.contact_forces[:, self.feet_indices, 2]/1.0).clip(min=0.,max=1.), # 4
                                              ), dim=-1)
        if self.num_privileged_obs is not None: # critic_obs, no noise
            self.privileged_obs_buf = torch.cat((self.estimator_input_buf, self.estimator_true_value), dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            # 121列
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.estimator_input_buf += (2 * torch.rand_like(self.estimator_input_buf) - 1) * self.noise_scale_vec
        # normal critic observation is with noise
        self.critic_obs_buf = torch.cat((self.estimator_input_buf, self.estimator_true_value), dim=-1)
    
    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.estimator_input_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = 0. # commands
        noise_vec[3:6] = noise_scales.gravity * noise_level * self.obs_scales.gravity
        noise_vec[6:9] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos # q_t
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel # qdot_t
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos # q_t-dt
        noise_vec[9+3*self.num_actions:9+4*self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos # q_t-2dt
        noise_vec[9+4*self.num_actions:9+5*self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos # q_t-3dt
        noise_vec[9+5*self.num_actions:9+6*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel # qdot_t-dt
        noise_vec[9+6*self.num_actions:9+7*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel # qdot_t-2dt
        noise_vec[9+7*self.num_actions:9+8*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel # qdot_t-3dt
        noise_vec[9+8*self.num_actions:9+9*self.num_actions] = 0. # a_t-1
        noise_vec[9+9*self.num_actions:9+10*self.num_actions] = 0. # a_t-2
        noise_vec[9+10*self.num_actions:9+10*self.num_actions+3*len(self.feet_names)] = noise_scales.dof_pos * noise_level
        noise_vec[141:145] = 0. # clock_input
        noise_vec[145:147] = 0. # phase_ratio
        # if self.cfg.terrain.measure_heights:
        #     noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
            
        return noise_vec
    
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)  # Periodic Reward Framework
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim) # Periodic Reward Framework
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)  # Periodic Reward Framework
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.base_quat = self.root_states[:, 3:7]
        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.foot_vel = self.rigid_body_states[:, self.feet_indices, 7:10]
        self.foot_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        # Periodic Reward Framework
        self.foot_vel_fl = self.rigid_body_states[:, self.foot_index_fl, 7:10]
        self.foot_vel_fr = self.rigid_body_states[:, self.foot_index_fr, 7:10]
        self.foot_vel_rl = self.rigid_body_states[:, self.foot_index_rl, 7:10]
        self.foot_vel_rr = self.rigid_body_states[:, self.foot_index_rr, 7:10]
        self.foot_pos_fl = self.rigid_body_states[:, self.foot_index_fl, 0:3]
        self.foot_pos_fr = self.rigid_body_states[:, self.foot_index_fr, 0:3]
        self.foot_pos_rl = self.rigid_body_states[:, self.foot_index_rl, 0:3]
        self.foot_pos_rr = self.rigid_body_states[:, self.foot_index_rr, 0:3]
        self.foot_contact_force_fl = self.contact_forces[:, self.foot_index_fl, :]
        self.foot_contact_force_fr = self.contact_forces[:, self.foot_index_fr, :]
        self.foot_contact_force_rl = self.contact_forces[:, self.foot_index_rl, :]
        self.foot_contact_force_rr = self.contact_forces[:, self.foot_index_rr, :]
        
        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.neutral_p_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.neutral_d_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros_like(self.actions)
        self.llast_actions = torch.zeros_like(self.last_actions)
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.p_b2f = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(len(self.feet_names)):
            self.p_b2f[:, 3*i:3*(i+1)] = quat_rotate_inverse(self.base_quat, self.foot_pos[:, i] - self.base_pos) # foot pos relative to base pos(base frame)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0
        # observation history
        self.last_dof_pos = torch.zeros_like(self.dof_pos)  # t-dt
        self.llast_dof_pos = torch.zeros_like(self.dof_pos) # t-2dt
        self.l3ast_dof_pos = torch.zeros_like(self.dof_pos) # t-3dt
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.llast_dof_vel = torch.zeros_like(self.dof_vel)
        self.l3ast_dof_vel = torch.zeros_like(self.dof_vel)
        # For OnPolicyRunnerEE
        self.estimator_input_buf = torch.zeros(self.num_envs, self.num_estimator_input, device=self.device, dtype=torch.float)
        self.critic_obs_buf = torch.zeros(self.num_envs, self.num_critic_obs, device=self.device, dtype=torch.float)
        self.noise_scale_vec = self._get_noise_scale_vec()
        #----- Periodic Reward Framework -----#
        self.gait_time = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.gait_period = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.phi = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.phase_ratio = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.b_swing = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.a_stance = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        if self.quad_enable:
            self.theta = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
            self.theta[:, 0] = self.cfg.rewards.periodic_reward_framework.quadruped.theta_fl[0] # default is the first value
            self.theta[:, 1] = self.cfg.rewards.periodic_reward_framework.quadruped.theta_fr[0]
            self.theta[:, 2] = self.cfg.rewards.periodic_reward_framework.quadruped.theta_rl[0]
            self.theta[:, 3] = self.cfg.rewards.periodic_reward_framework.quadruped.theta_rr[0]
            self.b_swing[:, :] = self.cfg.rewards.periodic_reward_framework.quadruped.b_swing[0] *2*torch.pi
            self.gait_period[:, :] = self.cfg.rewards.periodic_reward_framework.quadruped.gait_period
        self.a_stance = self.b_swing
        
        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.neutral_p_gains[:, i] = self.cfg.control.stiffness[dof_name]
                    self.neutral_d_gains[:, i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.neutral_p_gains[:, i] = 0.
                self.neutral_d_gains[:, i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        # randomize pd gains(when creating sim)
        if self.cfg.domain_rand.randomize_pd:
            self.p_gains = self.neutral_p_gains * torch.tensor(np.random.uniform(self.cfg.domain_rand.p_range[0], self.cfg.domain_rand.p_range[1], size=(self.num_envs, self.num_actions)), dtype=torch.float, device=self.device)
            self.d_gains = self.neutral_d_gains * torch.tensor(np.random.uniform(self.cfg.domain_rand.d_range[0], self.cfg.domain_rand.d_range[1], size=(self.num_envs, self.num_actions)), dtype=torch.float, device=self.device)
        else:
            self.p_gains = self.neutral_p_gains
            self.d_gains = self.neutral_d_gains
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
    
    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        # print("rigid_body_names: ", body_names)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        print(self.dof_names)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        self.feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        # print("feet_names: ", feet_names)
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(self.feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.feet_names[i])
        # Periodic Reward Framework. distinguish between 4 feet
        for i in range(len(self.feet_names)):
            if "FL" in self.feet_names[i]:
                self.foot_index_fl = self.feet_indices[i]
            elif "FR" in self.feet_names[i]:
                self.foot_index_fr = self.feet_indices[i]
            elif "RL" in self.feet_names[i]:
                self.foot_index_rl = self.feet_indices[i]
            elif "RR" in self.feet_names[i]:
                self.foot_index_rr = self.feet_indices[i]

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
    
    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        # Periodic Reward Framework. Constants are init here.
        self.quad_enable = self.cfg.rewards.periodic_reward_framework.quadruped.enable
        if self.quad_enable:
            self.a_swing = self.cfg.rewards.periodic_reward_framework.quadruped.a_swing * 2*torch.pi
            self.b_stance = self.cfg.rewards.periodic_reward_framework.quadruped.b_stance * 2*torch.pi
            self.selected_gait = self.cfg.rewards.periodic_reward_framework.quadruped.selected_gait
        self.kappa = self.cfg.rewards.periodic_reward_framework.kappa
        # get the sigmas for the reward functions
        self.tracking_sigma = self.cfg.rewards.tracking_sigma

        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.cfg.domain_rand.push_interval_rb = np.ceil(self.cfg.domain_rand.push_rb_interval_s / self.dt)
        
        # For OnPolicyRunnerEE
        self.num_estimator_input = self.cfg.env.num_estimator_input
        self.num_estimator_output = self.cfg.env.num_estimator_output
        self.num_critic_obs = self.cfg.env.num_critic_obs
        self.num_actor_obs = self.cfg.env.num_actor_obs
    
    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        
        if self.cfg.domain_rand.randomize_dof_friction:
            for j in range(self.num_dof):
                props["friction"][j] = torch.tensor(np.random.uniform(self.cfg.domain_rand.dof_friction_range[0], self.cfg.domain_rand.dof_friction_range[1]), dtype=torch.float, device=self.device)
        return props
    
    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        # randomize com position
        if self.cfg.domain_rand.randomize_com_pos:
            x_range = self.cfg.domain_rand.com_pos_x_range
            y_range = self.cfg.domain_rand.com_pos_y_range
            z_range = self.cfg.domain_rand.com_pos_z_range
            # randomize com position of "base1_downbox"
            props[0].com += gymapi.Vec3(np.random.uniform(x_range[0], x_range[1]), np.random.uniform(y_range[0], y_range[1]), np.random.uniform(z_range[0], z_range[1]))
            # print(f"com of base: {props[0].com} (after randomization)")
        return props
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.7 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, self.cfg.commands.min_curriculum_x, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum_x)
    
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_x_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        lin_vel_y_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        lin_vel_error = torch.sqrt(lin_vel_x_error + lin_vel_y_error)
        return torch.exp(-lin_vel_error/self.tracking_sigma) # for mode-aware function ablation study
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.tracking_sigma) # for mode-aware function ablation study
    
    #------------ Private Functions ----------------
    def calc_periodic_reward_obs(self):
        # calculate clock inputs for the policy
        num_cycle_timesteps = self.gait_period / self.dt
        if self.quad_enable:
            clock_input_fl = torch.sin(2*torch.pi*(self.phi+self.theta[:, 0].view(-1, 1)) / num_cycle_timesteps)
            clock_input_fr = torch.sin(2*torch.pi*(self.phi+self.theta[:, 1].view(-1, 1)) / num_cycle_timesteps)
            clock_input_rl = torch.sin(2*torch.pi*(self.phi+self.theta[:, 2].view(-1, 1)) / num_cycle_timesteps)
            clock_input_rr = torch.sin(2*torch.pi*(self.phi+self.theta[:, 3].view(-1, 1)) / num_cycle_timesteps)
            self.clock_input = torch.cat((clock_input_fl, clock_input_fr, clock_input_rl, clock_input_rr), dim=-1)
    
    def resample_phase_and_theta(self, env_ids):
        if self.quad_enable:
            if self.selected_gait is not None:
                gait = self.selected_gait
                if gait == "stand":
                    self.theta[env_ids, 0] = self.cfg.rewards.periodic_reward_framework.quadruped.theta_fl[0]
                    self.theta[env_ids, 1] = self.cfg.rewards.periodic_reward_framework.quadruped.theta_fr[0]
                    self.theta[env_ids, 2] = self.cfg.rewards.periodic_reward_framework.quadruped.theta_rl[0]
                    self.theta[env_ids, 3] = self.cfg.rewards.periodic_reward_framework.quadruped.theta_rr[0]
                    self.b_swing[env_ids, 0] = self.cfg.rewards.periodic_reward_framework.quadruped.b_swing[0] *2*torch.pi
                    self.a_stance = self.b_swing
                    self.phase_ratio[env_ids, 0] = self.cfg.rewards.periodic_reward_framework.quadruped.swing_phase_ratio[0]
                    self.phase_ratio[env_ids, 1] = self.cfg.rewards.periodic_reward_framework.quadruped.stance_phase_ratio[0]
                elif gait == "trot":
                    self.theta[env_ids, 0] = self.cfg.rewards.periodic_reward_framework.quadruped.theta_fl[1]
                    self.theta[env_ids, 1] = self.cfg.rewards.periodic_reward_framework.quadruped.theta_fr[1]
                    self.theta[env_ids, 2] = self.cfg.rewards.periodic_reward_framework.quadruped.theta_rl[1]
                    self.theta[env_ids, 3] = self.cfg.rewards.periodic_reward_framework.quadruped.theta_rr[1]
                    self.b_swing[env_ids, 0] = self.cfg.rewards.periodic_reward_framework.quadruped.b_swing[1] *2*torch.pi
                    self.a_stance = self.b_swing
                    self.phase_ratio[env_ids, 0] = self.cfg.rewards.periodic_reward_framework.quadruped.swing_phase_ratio[1]
                    self.phase_ratio[env_ids, 1] = self.cfg.rewards.periodic_reward_framework.quadruped.stance_phase_ratio[1]
            else:
                gait_list = [0, 1]
                gait_choice = np.random.choice(gait_list)
                # update theta
                self.theta[env_ids, 0] = self.cfg.rewards.periodic_reward_framework.quadruped.theta_fl[gait_choice]
                self.theta[env_ids, 1] = self.cfg.rewards.periodic_reward_framework.quadruped.theta_fr[gait_choice]
                self.theta[env_ids, 2] = self.cfg.rewards.periodic_reward_framework.quadruped.theta_rl[gait_choice]
                self.theta[env_ids, 3] = self.cfg.rewards.periodic_reward_framework.quadruped.theta_rr[gait_choice]
                # update b_swing, phase ratio
                self.b_swing[env_ids, 0] = self.cfg.rewards.periodic_reward_framework.quadruped.b_swing[gait_choice] *2*torch.pi
                self.a_stance = self.b_swing
                self.phase_ratio[env_ids, 0] = self.cfg.rewards.periodic_reward_framework.quadruped.swing_phase_ratio[gait_choice]
                self.phase_ratio[env_ids, 1] = self.cfg.rewards.periodic_reward_framework.quadruped.stance_phase_ratio[gait_choice]
    
    def _refresh_rigid_body_states(self):
        # refresh the states of the rigid bodies
        self.foot_vel = self.rigid_body_states[:, self.feet_indices, 7:10]
        self.foot_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        # Periodic Reward Framework
        self.foot_vel_fl = self.rigid_body_states[:, self.foot_index_fl, 7:10]
        self.foot_vel_fr = self.rigid_body_states[:, self.foot_index_fr, 7:10]
        self.foot_vel_rl = self.rigid_body_states[:, self.foot_index_rl, 7:10]
        self.foot_vel_rr = self.rigid_body_states[:, self.foot_index_rr, 7:10]
        self.foot_pos_fl = self.rigid_body_states[:, self.foot_index_fl, 0:3]
        self.foot_pos_fr = self.rigid_body_states[:, self.foot_index_fr, 0:3]
        self.foot_pos_rl = self.rigid_body_states[:, self.foot_index_rl, 0:3]
        self.foot_pos_rr = self.rigid_body_states[:, self.foot_index_rr, 0:3]
        self.foot_contact_force_fl = self.contact_forces[:, self.foot_index_fl, :]
        self.foot_contact_force_fr = self.contact_forces[:, self.foot_index_fr, :]
        self.foot_contact_force_rl = self.contact_forces[:, self.foot_index_rl, :]
        self.foot_contact_force_rr = self.contact_forces[:, self.foot_index_rr, :]
    
    def update_curriculum(self, env_ids):
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        # update every max_episode_length
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
    
    #------------ Private Reward Functions ----------------
    def uniped_periodic_reward(self, foot_type):
        # coefficient
        c_swing_spd = 0 # speed is not penalized during swing phase
        c_swing_frc = -1 # force is penalized during swing phase
        c_stance_spd = -1 # speed is penalized during stance phase
        c_stance_frc = 0 # force is not penalized during stance phase
        
        if self.quad_enable:
            # q_frc and q_spd
            if foot_type == "FL":
                q_frc = torch.norm(self.foot_contact_force_fl, dim=-1).view(-1, 1)
                q_spd = torch.norm(self.foot_vel_fl, dim=-1).view(-1, 1)
                # size: num_envs; need to reshape to (num_envs, 1), or there will be error due to broadcasting
                
                phi = (self.phi + self.theta[:, 0].view(-1, 1)) % 1.0 # modulo phi over 1.0 to get cicular phi in [0, 1.0]
            elif foot_type == "FR":
                q_frc = torch.norm(self.foot_contact_force_fr, dim=-1).view(-1, 1)
                q_spd = torch.norm(self.foot_vel_fr, dim=-1).view(-1, 1)
                phi = (self.phi + self.theta[:, 1].view(-1, 1)) % 1.0 # modulo phi over 1.0 to get cicular phi in [0, 1.0]
            elif foot_type == "RL":
                q_frc = torch.norm(self.foot_contact_force_rl, dim=-1).view(-1, 1)
                q_spd = torch.norm(self.foot_vel_rl, dim=-1).view(-1, 1)
                phi = (self.phi + self.theta[:, 2].view(-1, 1)) % 1.0 # modulo phi over 1.0 to get cicular phi in [0, 1.0]
            elif foot_type == "RR":
                q_frc = torch.norm(self.foot_contact_force_rr, dim=-1).view(-1, 1)
                q_spd = torch.norm(self.foot_vel_rr, dim=-1).view(-1, 1)
                phi = (self.phi + self.theta[:, 3].view(-1, 1)) % 1.0 # modulo phi over 1.0 to get cicular phi in [0, 1.0]
    
        phi *= 2 * torch.pi # convert phi to radians
        # clip the value of phi to [0, 1.0]. The vonmises function in scipy may return cdf outside [0, 1.0]
        F_A_swing = torch.clip(torch.tensor(vonmises.cdf(loc=self.a_swing.cpu(), kappa=self.kappa, x=phi.cpu()), device=self.device), 0.0, 1.0)
        F_B_swing = torch.clip(torch.tensor(vonmises.cdf(loc=self.b_swing.cpu(), kappa=self.kappa, x=phi.cpu()), device=self.device), 0.0, 1.0)
        F_A_stance = torch.clip(torch.tensor(vonmises.cdf(loc=self.a_stance.cpu(), kappa=self.kappa, x=phi.cpu()), device=self.device), 0.0, 1.0)
        F_B_stance = torch.clip(torch.tensor(vonmises.cdf(loc=self.b_stance.cpu(), kappa=self.kappa, x=phi.cpu()), device=self.device), 0.0, 1.0)
        
        # calc the expected C_spd and C_frc according to the formula in the paper
        exp_swing_ind = F_A_swing * (1 - F_B_swing)
        exp_stance_ind = F_A_stance * (1 - F_B_stance)
        exp_C_spd_ori = c_swing_spd * exp_swing_ind + c_stance_spd * exp_stance_ind
        exp_C_frc_ori = c_swing_frc * exp_swing_ind + c_stance_frc * exp_stance_ind
        
        # just the code above can't result in the same reward curve as the paper
        # a little trick is implemented to make the reward curve same as the paper
        # first let all envs get the same exp_C_frc and exp_C_spd
        exp_C_frc = -0.5 + (-0.5 - exp_C_spd_ori) 
        exp_C_spd = exp_C_spd_ori
        # select the envs that are in swing phase
        is_in_swing = (phi >= self.a_swing) & (phi < self.b_swing)
        indices_in_swing = is_in_swing.nonzero(as_tuple=False).flatten()
        # update the exp_C_frc and exp_C_spd of the envs in swing phase
        exp_C_frc[indices_in_swing] = exp_C_frc_ori[indices_in_swing]
        exp_C_spd[indices_in_swing] = -0.5 + (-0.5 - exp_C_frc_ori[indices_in_swing])
        
        # Judge if it's the standing gait
        is_standing = (self.b_swing[:] == self.a_swing).nonzero(as_tuple=False).flatten()
        exp_C_frc[is_standing] = 0
        exp_C_spd[is_standing] = -1
        
        
        return exp_C_spd * q_spd + exp_C_frc * q_frc, exp_C_spd, exp_C_frc
    
    def _reward_periodic_reward(self):
        if self.quad_enable:
            # reward for each foot
            reward_fl, self.exp_C_spd_fl, self.exp_C_frc_fl = self.uniped_periodic_reward("FL")
            reward_fr, self.exp_C_spd_fr, self.exp_C_frc_fr = self.uniped_periodic_reward("FR")
            reward_rl, self.exp_C_spd_rl, self.exp_C_frc_rl = self.uniped_periodic_reward("RL")
            reward_rr, self.exp_C_spd_rr, self.exp_C_frc_rr = self.uniped_periodic_reward("RR")
            # reward for the whole body
            reward = reward_fl.flatten() + reward_fr.flatten() + reward_rl.flatten() + reward_rr.flatten()
        return torch.exp(reward / 2.0)
    
    def _reward_action_smoothness(self):
        '''Penalize action smoothness'''
        action_smoothness_cost = torch.sum(torch.square(self.actions - 2*self.last_actions + self.llast_actions), dim=-1)
        return action_smoothness_cost
    
    def _reward_foot_clearance(self):
        '''reward for foot clearance'''
        foot_vel_xy_norm = torch.norm(self.foot_vel[:, :, [0, 1]], dim=-1)
        reward = torch.sum(foot_vel_xy_norm * torch.square(self.foot_pos[:, :, 2] - self.cfg.rewards.foot_clearance_target), dim=-1)
        return torch.exp(-reward/self.cfg.rewards.foot_clearance_tracking_sigma) # positive formulation can learn better
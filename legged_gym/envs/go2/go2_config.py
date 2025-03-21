from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import torch

class GO2Cfg( LeggedRobotCfg ):
    seed = 0
    class env( LeggedRobotCfg.env ):
        num_envs = 8192
        num_obs = 153
        num_privileged_obs = num_obs
        num_actions = 12
        episode_length_s = 20  # episode length in seconds
    
    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = "plane"
        border_size = 25    # [m]
        curriculum = True   # 用heightfield时curriculum会行列颠倒，但trimesh没问题
        static_friction = 1.0
        dynamic_friction = 1.0
        # rough terrain only:
        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False      # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        num_rows= 10  # number of terrain rows (levels)
        num_cols = 10 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
        
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': 0.0 ,  # [rad]
            'RR_hip_joint': 0.0,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 0.8,   # [rad]  # test whether it can stand on rear legs
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 0.8,   # [rad]
            'RR_thigh_joint': 0.8,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }
        randomize_base_orientation = True
        init_state_train = True
        init_state_play = False

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4  # policy frequency 50Hz

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = []
        terminate_after_contacts_on = ["base", 'Head', "calf", "thigh"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        fix_base_link = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.36
        standing_height_target = 0.40
        tracking_sigma = 0.25
        point_tracking_sigma = 0.1
        base_height_upright_target = 0.50
        base_angle_upright_target = 1.2
        only_positive_rewards = True
        class scales( LeggedRobotCfg.rewards.scales ):
            # limitation
            termination = -200.0
            dof_pos_limits = -5.0
            # command tracking
            tracking_lin_vel = 4.0 # 4.0
            tracking_ang_vel = 4.0 # 4.0
            # smooth
            lin_vel_x = -2.0
            ang_vel_xy = -0.05   # -0.01
            dof_vel = -5.e-4
            dof_acc = -2.e-7
            action_rate = -1.e-2
            action_smoothness = -5.e-3
            torques = -2.e-4
            
            # upright/quad2upright
            # lower_legs_cost = -0.5
            lower_legs_tracking = 2.0
            base_angle_upright = 4.0
            base_height_upright = 4.0
            hip_close_to_default = -1.0
            upright_contacts = 2.0
            
            # gait
            periodic_reward = 4.0 # 5.0
            # dof_close_to_default = -1.0
        
        class periodic_reward_framework:
            '''Periodic reward framework in OSU's paper(https://arxiv.org/abs/2011.01387)'''
            kappa = 20
            resampling_time = 4.0                     # resampling time [s], gait transition needs to resample more often than the command transition
            class quadruped:
                enable = False
                gait_period = 0.4                         # gait period [s]
                num_gaits = 3                             # [standing, walking, trotting]
                selected_gait = "trot"
                a_swing = torch.tensor(0.0)               # start of swing is all the same
                b_swing = torch.tensor([0.0, 0.4])
                a_stance = b_swing
                b_stance = torch.tensor(1.0)
                theta_fl = torch.tensor([0.0, 0.0])  # front left leg
                theta_fr = torch.tensor([0.0, 0.5])
                theta_rl = torch.tensor([0.0, 0.5])  # rear left leg
                theta_rr = torch.tensor([0.0, 0.0])
                swing_phase_ratio = b_swing - a_swing
                stance_phase_ratio = 1 - swing_phase_ratio
            class biped:
                enable = True
                gait_period = 0.35                         # gait period [s]
                num_gaits = 1                             # [walking]
                selected_gait = None
                a_swing = torch.tensor(0.0)               # start of swing is all the same
                b_swing = torch.tensor([0.4])
                a_stance = b_swing
                b_stance = torch.tensor(1.0)
                theta_left = torch.tensor([0.0])
                theta_right = torch.tensor([0.5])
                swing_phase_ratio = b_swing - a_swing
                stance_phase_ratio = 1 - swing_phase_ratio
    
    class commands( LeggedRobotCfg.commands ):
        curriculum = False
        max_curriculum_x = 1.5
        min_curriculum_x = -1.0
        num_commands = 4       # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 6.   # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges( LeggedRobotCfg.commands.ranges ):
            # lin_vel_x = [-0.5, 0.5]      # min max [m/s]
            # lin_vel_y = [-1.0, 1.0]      # min max [m/s]
            # ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            # heading = [-3.14, 3.14]
            
            lin_vel_x = [-0.5, 0.5]   # min max [m/s]
            lin_vel_y = [0.0, 0.0]    # min max [m/s]
            ang_vel_yaw = [0.0, 0.0]     # min max [rad/s]
            heading = [-3.14, 3.14]
        class tracking_points:
            left_limb = [-0.2, 0.15, -0.3]
            right_limb = [-0.2, -0.15, -0.3]
            randomization = False
            randomize_range = [-0.05, 0.05]
    
    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_friction = True
        friction_range = [0.5, 1.5]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True            # perturbance force
        push_interval_s = 10
        max_push_vel_xy = 0.5
        randomize_pd = False
        p_range = [-2, 2]
        d_range = [-0.05, 0.05]
        randomize_dof_friction = False
        dof_friction_range = [0.0, 0.2]
        push_rigid_bodies = False
        max_push_force = 200.
        push_rb_interval_s = 15
        randomize_restitution = True
        restitution_range = [0.0, 0.5]
    
    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25 # 
            dof_pos = 1.0  # 
            dof_vel = 0.05
            rpy = 1.0
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.
    
    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            rpy = 0.05
            height_measurements = 0.1
    
    class viewer( LeggedRobotCfg.viewer ):
        ref_env = 0
        pos = [0, -4, 3]  # [m]
        lookat = [0., 0, 1.]  # [m]
        debug_viz = False
    
    class sim( LeggedRobotCfg.sim ):
        dt =  0.005  # 200Hz
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

class GO2CfgPPO( LeggedRobotCfgPPO ):
    seed = 0
    runner_class_name = 'OnPolicyRunner'
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        learning_rate = 1.e-3 #5.e-4
        # symmetry_loss_coef = 4.0
        num_mini_batches = 4
    class policy( LeggedRobotCfgPPO.policy ):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCritic'
        run_name = ''
        num_steps_per_env = 24 # per iteration
        save_interval = 100 # check for potential saves every this many iterations
        experiment_name = 'go2_q2b'
        load_run = "Dec30_18-53-43_" # -1 = last run
        checkpoint = 800 # -1 = last saved model
        max_iterations = 2500 # number of policy updates
    
    # class estimator():
    #     hidden_dims = [256, 128]
    #     activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    #     num_mini_batches = 4
    #     max_epochs = 5
    #     lr = 1.e-3
    
    # class estimator_lr_scheduler:
    #     type = None # could be LinearLR
    #     # for LinearLR(https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html)
    #     start_factor = 1.0
    #     end_factor = 0.1
    #     total_iters = 1000

    # class symmetry:
    #     enable = False
    #     mirror_indices={
    #         ## 考虑髋,膝关节的对称
    #         ### observation:
    #         "com_obs_inds": [0,2,4,7],
    #         "neg_obs_inds": [1,3,5,6,8],
    #         "left_obs_inds": list(range(12,15))+list(range(18,21)) + list(range(24,27))+list(range(30,33)) + list(range(36,39))+list(range(42,45)),
    #         "right_obs_inds":list(range(15,18))+list(range(21,24)) + list(range(27,30))+list(range(33,36)) + list(range(39,42))+list(range(45,48)),
    #         "sideneg_obs_inds":[],
    #         #### action:
    #         "com_act_inds": [],
    #         "neg_act_inds": [],
    #         "sideneg_act_inds":[],
    #         "left_act_inds": list(range(0, 3)) + list(range(6, 9)), # FL, RL
    #         "right_act_inds": list(range(3, 6)) + list(range(9, 12)), # FR, RR
    #     }

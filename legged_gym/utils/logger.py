import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value
import matplotlib.pyplot as plt
import xlsxwriter

# 1. Specify the excel filename and directory
EXCEL_FILENAME = "/home/jason/Documents/2024/little_biped_test/excel_data/20250319_test_sim2real.xlsx"

class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()
    
    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()   
    
    def set_header_of_xlsx(self):
        self.workbook = xlsxwriter.Workbook(EXCEL_FILENAME)
        self.worksheet = self.workbook.add_worksheet()
        label = list(self.state_log.keys())
        for i in range(len(label)):
            self.worksheet.write(0, i, label[i])
            
    def save_data_to_xlsx(self):
        '''
        save the data to a excel file
        '''
        # set header
        self.set_header_of_xlsx()
        # get the first key, to get the length of the data
        first_key = list(self.state_log.keys())[0]
        for row in range(len(self.state_log[first_key])):
            for col, key in enumerate(self.state_log.keys()):
                self.worksheet.write(1+row, col, self.state_log[key][row])
        self.workbook.close()
        print("xlsx file created and filled!")
    
    def _plot(self):
        
        nb_rows = 4
        nb_cols = 5
        fig, axs = plt.subplots(nb_rows, nb_cols, tight_layout=True)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        # plot GRF in row 0
        a = axs[0, 0]
        if log["GRF_left"]: a.plot(time, log["GRF_left"], label='left')
        if log["GRF_fl"]: a.plot(time, log["GRF_fl"], label='fl')
        a.set(xlabel='time [s]', ylabel='GRF [N]', title='quad GRF')
        a.legend()
        a = axs[0, 1]
        if log["GRF_right"]: a.plot(time, log["GRF_right"], label='right')
        if log["GRF_fr"]: a.plot(time, log["GRF_fr"], label='fr')
        a.set(xlabel='time [s]', ylabel='GRF [N]', title='quad GRF')
        a.legend()
        # a = axs[0, 2]
        # if log["GRF_rl"]: a.plot(time, log["GRF_rl"], label='rl')
        # a.set(xlabel='time [s]', ylabel='GRF [N]', title='quad GRF')
        # a.legend()
        # a = axs[0, 3]
        # if log["GRF_rr"]: a.plot(time, log["GRF_rr"], label='rr')
        # a.set(xlabel='time [s]', ylabel='GRF [N]', title='quad GRF')
        # a.legend()

        # plot E[C_frc] in row 1
        a = axs[1, 0]
        if log["E[C_frc_left]"]: a.plot(time, log["E[C_frc_left]"], label='left')
        if log["E[C_frc_fl]"]: a.plot(time, log["E[C_frc_fl]"], label='fl')
        a.set(xlabel='time [s]', ylabel='E[C]', title='quad E[C]')
        a.legend()
        a = axs[1, 1]
        if log["E[C_frc_right]"]: a.plot(time, log["E[C_frc_right]"], label='right')
        if log["E[C_frc_fr]"]: a.plot(time, log["E[C_frc_fr]"], label='fr')
        a.set(xlabel='time [s]', ylabel='E[C]', title='quad E[C]')
        a.legend()
        # a = axs[1, 2]
        # if log["E[C_frc_rl]"]: a.plot(time, log["E[C_frc_rl]"], label='rl')
        # a.set(xlabel='time [s]', ylabel='E[C]', title='quad E[C]')
        # a.legend()
        # a = axs[1, 3]
        # if log["E[C_frc_rr]"]: a.plot(time, log["E[C_frc_rr]"], label='rr')
        # a.set(xlabel='time [s]', ylabel='E[C]', title='quad E[C]')
        # a.legend()
        
        # plt vel_cmd and com_vel_x
        a = axs[3, 0]
        if log['base_lin_vel_x']: a.plot(time, log['base_lin_vel_x'], label='true')
        if log["est_lin_vel_x"]: a.plot(time, log["est_lin_vel_x"], label='estimated')
        if log['vel_cmd_x']: a.plot(time, log['vel_cmd_x'], label='command')
        if log["base_ang_vel_x"]: a.plot(time, log["base_ang_vel_x"], label='base_ang_vel_x')
        if log["base_roll"]: a.plot(time, log["base_roll"], label='roll')
        a.set(xlabel='time [s]', ylabel='vel_x [m/s]', title='Velocity x')
        a.legend()
        a = axs[3, 1]
        if log['base_lin_vel_y']: a.plot(time, log['base_lin_vel_y'], label='true')
        if log["est_lin_vel_y"]: a.plot(time, log["est_lin_vel_y"], label='estimated')
        if log['vel_cmd_y']: a.plot(time, log['vel_cmd_y'], label='command')
        if log["base_ang_vel_y"]: a.plot(time, log["base_ang_vel_y"], label='base_ang_vel_y')
        if log["base_pitch"]: a.plot(time, log["base_pitch"], label='pitch')
        a.set(xlabel='time [s]', ylabel='vel_y [m/s]', title='Velocity y')
        a.legend()
        a = axs[3, 2]
        if log['vel_cmd_yaw']: a.plot(time, log['vel_cmd_yaw'], label='vel_cmd_yaw')
        if log['base_ang_vel_z']: a.plot(time, log['base_ang_vel_z'], label='base_ang_vel_z')
        if log["base_lin_vel_z"]: a.plot(time, log["base_lin_vel_z"], label='true_lin_vel_z')
        if log["est_lin_vel_z"]: a.plot(time, log["est_lin_vel_z"], label='estimated_lin_vel_z')
        if log["base_yaw"]: a.plot(time, log["base_yaw"], label='yaw')
        a.set(xlabel='time [s]', ylabel='vel_z [rad/s]', title='Velocity z')
        a.legend()
        
        # torques
        a = axs[2, 0]
        if log['left_hip_torque']: a.plot(time, log['left_hip_torque'], label='left_hip_torque')
        if log["right_hip_torque"]: a.plot(time, log["right_hip_torque"], label='right_hip_torque')
        a.set(xlabel='time [s]', ylabel='hip_torque [Nm]', title='hip_torque')
        a.legend()
        a = axs[2, 1]
        if log['left_thigh_torque']: a.plot(time, log['left_thigh_torque'], label='left_thigh_torque')
        if log["right_thigh_torque"]: a.plot(time, log["right_thigh_torque"], label='right_thigh_torque')
        a.set(xlabel='time [s]', ylabel='thigh_torque [Nm]', title='thigh_torque')
        a.legend()
        a = axs[2, 2]
        if log['left_calf_torque']: a.plot(time, log['left_calf_torque'], label='left_calf_torque')
        if log["right_calf_torque"]: a.plot(time, log["right_calf_torque"], label='right_calf_torque')
        a.set(xlabel='time [s]', ylabel='calf_torque [Nm]', title='calf_torque')
        a.legend()
        
        # true/estimated foot height
        # a = axs[0, 0]
        # if log["true_fl_foot_height"]: a.plot(time, log["true_fl_foot_height"], label='true')
        # if log["estimated_fl_foot_height"]: a.plot(time, log["estimated_fl_foot_height"], label='estimated')
        # a.set(xlabel='time [s]', ylabel='foot_height [m]', title='foot_height')
        # a.legend()
        # a = axs[0, 1]
        # if log["true_fr_foot_height"]: a.plot(time, log["true_fr_foot_height"], label='true')
        # if log["estimated_fr_foot_height"]: a.plot(time, log["estimated_fr_foot_height"], label='estimated')
        # a.set(xlabel='time [s]', ylabel='foot_height [m]', title='foot_height')
        # a.legend()
        # a = axs[0, 2]
        # if log["true_rl_foot_height"]: a.plot(time, log["true_rl_foot_height"], label='true')
        # if log["estimated_rl_foot_height"]: a.plot(time, log["estimated_rl_foot_height"], label='estimated')
        # a.set(xlabel='time [s]', ylabel='foot_height [m]', title='foot_height')
        # a.legend()
        # a = axs[0, 3]
        # if log["true_rr_foot_height"]: a.plot(time, log["true_rr_foot_height"], label='true')
        # if log["estimated_rr_foot_height"]: a.plot(time, log["estimated_rr_foot_height"], label='estimated')
        # a.set(xlabel='time [s]', ylabel='foot_height [m]', title='foot_height')
        # a.legend()
        
        # true/estimated contact probability
        # a = axs[0, 4]
        # if log["true_fl_contact_prob"]: a.plot(time, log["true_fl_contact_prob"], label='true')
        # if log["estimated_fl_contact_prob"]: a.plot(time, log["estimated_fl_contact_prob"], label='estimated')
        # a.set(xlabel='time [s]', ylabel='contact_prob', title='contact_prob')
        # a.legend()
        # a = axs[1, 0]
        # if log["true_fr_contact_prob"]: a.plot(time, log["true_fr_contact_prob"], label='true')
        # if log["estimated_fr_contact_prob"]: a.plot(time, log["estimated_fr_contact_prob"], label='estimated')
        # a.set(xlabel='time [s]', ylabel='contact_prob', title='contact_prob')
        # a.legend()
        # a = axs[1, 1]
        # if log["true_rl_contact_prob"]: a.plot(time, log["true_rl_contact_prob"], label='true')
        # if log["estimated_rl_contact_prob"]: a.plot(time, log["estimated_rl_contact_prob"], label='estimated')
        # a.set(xlabel='time [s]', ylabel='contact_prob', title='contact_prob')
        # a.legend()
        # a = axs[1, 2]
        # if log["true_rr_contact_prob"]: a.plot(time, log["true_rr_contact_prob"], label='true')
        # if log["estimated_rr_contact_prob"]: a.plot(time, log["estimated_rr_contact_prob"], label='estimated')
        # a.set(xlabel='time [s]', ylabel='contact_prob', title='contact_prob')
        # a.legend()
        
        # action and dof_pos
        a = axs[0, 2]
        if log["desired_left_hip_dof_pos"]: a.plot(time, log["desired_left_hip_dof_pos"], label="desired pos")
        if log["jpos_left_hip"]: a.plot(time, log["jpos_left_hip"], label='actual pos')
        a.set(xlabel='time [s]', ylabel='left hip pos [rad]', title='Position')
        a.legend()
        a = axs[0, 3]
        if log["desired_left_thigh_dof_pos"]: a.plot(time, log["desired_left_thigh_dof_pos"], label="desired pos")
        if log["jpos_left_thigh"]: a.plot(time, log["jpos_left_thigh"], label='actual pos')
        a.set(xlabel='time [s]', ylabel='left thigh pos [rad]', title='Position')
        a.legend()
        a = axs[0, 4]
        if log["desired_left_calf_dof_pos"]: a.plot(time, log["desired_left_calf_dof_pos"], label="desired pos")
        if log['jpos_left_calf']: a.plot(time, log['jpos_left_calf'], label='actual pos')
        a.set(xlabel='time [s]', ylabel='left calf pos [rad]', title='Position')
        a.legend()
        a = axs[1, 2]
        if log["desired_right_hip_dof_pos"]: a.plot(time, log["desired_right_hip_dof_pos"], label="desired pos")
        if log["jpos_right_hip"]: a.plot(time, log["jpos_right_hip"], label='actual pos')
        a.set(xlabel='time [s]', ylabel='right hip pos [rad]', title='Position')
        a.legend()
        a = axs[1, 3]
        if log["desired_right_thigh_dof_pos"]: a.plot(time, log["desired_right_thigh_dof_pos"], label="desired pos")
        if log["jpos_right_thigh"]: a.plot(time, log["jpos_right_thigh"], label='actual pos')
        a.set(xlabel='time [s]', ylabel='right thigh pos [rad]', title='Position')
        a.legend()
        a = axs[1, 4]
        if log["desired_right_calf_dof_pos"]: a.plot(time, log["desired_right_calf_dof_pos"], label="desired pos")
        if log["jpos_right_calf"]: a.plot(time, log["jpos_right_calf"], label='actual pos')
        a.set(xlabel='time [s]', ylabel='right calf pos [rad]', title='Position')
        a.legend()
        
        plt.show()

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()
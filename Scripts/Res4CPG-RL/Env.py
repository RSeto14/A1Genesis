import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
import os
import random
import numpy as np
import re
import json
from rsl_rl.runners import OnPolicyRunner

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir_1 = os.path.dirname(script_dir) # DRL_Genesis/Scripts
parent_dir_2 = os.path.dirname(parent_dir_1) # DRL_Genesis

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower




class A1Res4CPGEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="cuda",capture=False, eval=False):
        self.device = torch.device(device)
        self.capture = capture
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.eval = eval
        self.last_gain_reset = 0 # [s]
        self.last_friction_reset = 0 # [s]

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        
        self.reward_scales = reward_cfg["reward_scales"]

        self.cpg_log_dirs = []
        for log in env_cfg["logs"]:
            self.cpg_log_dirs.append(os.path.join(parent_dir_1, "CPG-RL", "Logs",log))

        self.cpg_model_paths = []
        for cpg_log_dir in self.cpg_log_dirs:
            network_dir = os.path.join(cpg_log_dir, "Networks")
            if not os.path.exists(network_dir):
                print(f"Network directory not found: {network_dir}")
                continue
            model_files = [f for f in os.listdir(network_dir) if re.match(r"model_\d+\.pt", f)]
            if not model_files:
                print(f"No model files found in: {network_dir}")
                continue
            # 数字部分を抽出して最大のものを探す
            max_model = max(model_files, key=lambda f: int(re.search(r"\d+", f).group()))
            self.cpg_model_paths.append(os.path.join(network_dir, max_model))

        self.cpg_train_cfgs = []
        self.cpg_env_cfgs = []
        self.cpg_obs_cfgs = []
        self.cpg_commands_cfgs = []
        for cpg_log_dir in self.cpg_log_dirs:
            with open(f"{cpg_log_dir}/config.json", 'r') as json_file:
                loaded_config = json.load(json_file)

            self.cpg_train_cfgs.append(loaded_config["train_cfg"])
            self.cpg_env_cfgs.append(loaded_config["env_cfg"])
            self.cpg_obs_cfgs.append(loaded_config["obs_cfg"])
            self.cpg_commands_cfgs.append(loaded_config["command_cfg"])
        self.cpg_env = A1CPGEnv(
        num_envs=self.num_envs,
        env_cfg=self.cpg_env_cfgs[0],
        obs_cfg=self.cpg_obs_cfgs[0],
        device=self.device
        )

        self.cpg_obs_scales_list = [self.cpg_obs_cfgs[i]["obs_scales"] for i in range(len(self.cpg_env_cfgs))]
        self.cpg_obs_scales = self.cpg_obs_scales_list[0]
        # self.cpg_commands_scale_list = [torch.tensor(
        #     [self.cpg_obs_scales_list[i]["lin_vel"], self.cpg_obs_scales_list[i]["lin_vel"], self.cpg_obs_scales_list[i]["ang_vel"]],
        #     device=self.device,
        #     dtype=gs.tc_float,
        # ) for i in range(len(self.cpg_env_cfgs))]


        # self.cpg_default_dof_pos_list = [
        #     torch.tensor([self.cpg_env_cfgs[i]["default_joint_angles"][name] for name in self.cpg_env_cfgs[i]["dof_names"]],device=self.device,dtype=gs.tc_float) 
        #     for i in range(len(self.cpg_env_cfgs))]


        self.cpg_policy = []
        for i in range(len(self.cpg_log_dirs)):
            runner = OnPolicyRunner(self.cpg_env, self.cpg_train_cfgs[i], self.cpg_log_dirs[i], device=self.device)
            runner.load(self.cpg_model_paths[i])
            self.cpg_policy.append(runner.get_inference_policy(device=self.device))

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=num_envs, plane_reflection=False,),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
            
        )



        if self.capture:
            self.cam = self.scene.add_camera(
                res=(640, 480),
                # res=(1920, 1080),
                pos=(2.0, 1.0, 1.5),
                lookat=(0.0, 0.0, 0.5),
                up=(0.0, 0.0, 1.0),
                fov=40,
            aperture=0.0,
            # focus_dist=1.0,
            GUI=False,
            spp=1,
            denoise=True,
            )



        # add terrain
        # self.scene.add_entity(gs.morphs.MJCF(file="/mnt/ssd1/ryosei/master/Genesis_WS/Description/unitree_go2/flat.xml",visualization=True))


        # self.terrain = self.scene.add_entity(gs.morphs.Terrain(n_subterrains=self.env_cfg["n_subterrains"],subterrain_types=self.env_cfg["subterrain_types"],subterrain_size=self.env_cfg["subterrain_size"],horizontal_scale=self.env_cfg["horizontal_scale"],vertical_scale=self.env_cfg["vertical_scale"],visualization=True))

        # self.height_field = self.terrain.geoms[0].metadata["height_field"]
        
        # self.terrain = self.scene.add_entity(gs.morphs.Terrain(n_subterrains=(1,1),subterrain_types=[["fractal_terrain"]],subterrain_size=(10,10),horizontal_scale=0.25,vertical_scale=0.005,visualization=True))
        # self.terrain =self.scene.add_entity(gs.morphs.Plane(visualization=True))
        self.terrain =self.scene.add_entity(gs.morphs.URDF(file=f"{parent_dir_2}/Description/plane/plane.urdf",pos=[0,0,0], fixed=True, visualization=True))
        
        # self.height_field = torch.zeros((80,160), device=self.device, dtype=gs.tc_float)

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file=f"{parent_dir_2}/Description/A1/a1.xml",
                # pos=self.base_init_pos.cpu().numpy(),
                # quat=self.base_init_quat.cpu().numpy(),
            ),
            
        )
        # self.robot = self.scene.add_entity(
        #     gs.morphs.URDF(
        #         file="/mnt/ssd1/ryosei/master/Genesis_WS/Description/Go2/urdf/go2.urdf",
        #         pos=self.base_init_pos.cpu().numpy(),
        #         quat=self.base_init_quat.cpu().numpy(),
        #     ),
        # )

        self.put_box(n_row=self.env_cfg["n_row"],n_col=self.env_cfg["n_col"],box_size=self.env_cfg["box_size"],box_height_range=self.env_cfg["box_height_range"])

        # build
        self.scene.build(n_envs=num_envs,env_spacing=(0,0))

        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]

        # PD control parameters
        self._reset_pd_params()


        

        self.calf_link_idx = self.get_idx_by_name(["FR_calf", "FL_calf", "RR_calf", "RL_calf"])

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_lin_acc = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.obs_buf_cpg = torch.zeros((self.num_envs, self.cpg_obs_cfgs[0]["num_obs"]), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.cpg_commands_scale = torch.tensor(
            [self.cpg_obs_scales["lin_vel"], self.cpg_obs_scales["lin_vel"], self.cpg_obs_scales["ang_vel"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros((self.num_envs, 12), device=self.device, dtype=gs.tc_float)
        self.dof_pos = torch.zeros((self.num_envs, 12), device=self.device, dtype=gs.tc_float)
        self.dof_vel = torch.zeros((self.num_envs, 12), device=self.device, dtype=gs.tc_float)
        self.last_dof_vel = torch.zeros((self.num_envs, 12), device=self.device, dtype=gs.tc_float)
        self.dof_force = torch.zeros((self.num_envs, 12), device=self.device, dtype=gs.tc_float)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.foot_contact = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_int)
        self.foot_forces = torch.zeros((self.num_envs, 4, 3), device=self.device, dtype=gs.tc_float)
        self.foot_force_norms = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.extras = dict()  # extra information for logging

        # CPG parameters
        self.a = self.env_cfg["a"]
        self.d = self.env_cfg["d"]
        self.h = torch.tensor(self.env_cfg["h"][0], device=self.device, dtype=gs.tc_float).repeat(self.num_envs, 4)
        self.gc = torch.tensor(self.env_cfg["gc"][0], device=self.device, dtype=gs.tc_float).repeat(self.num_envs, 4)
        self.gp = torch.tensor(self.env_cfg["gp"][0], device=self.device, dtype=gs.tc_float).repeat(self.num_envs, 4)

        self.mu = torch.ones((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.omega = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.psi = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.r = torch.ones((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.r_dot = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.r_ddot = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        # self.theta = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.theta = torch.tensor([0, torch.pi, torch.pi, 0], device=self.device, dtype=gs.tc_float).repeat(self.num_envs, 1)
        self.phi = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)

        self.noise_std_dof_pos = self.obs_cfg["noise_std_dof_pos"]  # dof_posのノイズ標準偏差
        self.noise_std_dof_vel = self.obs_cfg["noise_std_dof_vel"]  # dof_velのノイズ標準偏差
        self.noise_std_euler = self.obs_cfg["noise_std_euler"]    # eulerのノイズ標準偏差
        self.noise_std_ang_vel = self.obs_cfg["noise_std_ang_vel"]  # ang_velのノイズ標準偏差

        self.last_cpg_actions = torch.zeros((self.num_envs, 12), device=self.device, dtype=gs.tc_float)
        self.cpg_actions = torch.zeros((self.num_envs, 12), device=self.device, dtype=gs.tc_float)
        # self.policy_idx = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.policy_idx = torch.arange(self.num_envs, device=self.device) % len(self.cpg_policy)
        if self.env_cfg["pd_stop"]:
            self.pd_stop_idx = torch.arange(0, self.num_envs, len(self.cpg_policy) * self.env_cfg["pd_stop_interval"], device=self.device)

        self.action_scale_mu = torch.tensor(
            [[self.cpg_env_cfgs[self.policy_idx[i]]["action_scale_mu"]] * 4 for i in range(self.num_envs)],
            device=self.device
        )

        self.action_scale_omega = torch.tensor(
            [[self.cpg_env_cfgs[self.policy_idx[i]]["action_scale_omega"]] * 4 for i in range(self.num_envs)],
            device=self.device
        )

        self.action_scale_psi = torch.tensor(
            [[self.cpg_env_cfgs[self.policy_idx[i]]["action_scale_psi"]] * 4 for i in range(self.num_envs)],
            device=self.device
        )

        # self.h = torch.tensor([[self.cpg_env_cfgs[self.policy_idx[i]]["h"][0]] * 4 for i in range(self.num_envs)], device=self.device, dtype=gs.tc_float)
        # self.gc = torch.tensor([[self.cpg_env_cfgs[self.policy_idx[i]]["gc"][0]] * 4 for i in range(self.num_envs)], device=self.device, dtype=gs.tc_float)
        # self.gp = torch.tensor([[self.cpg_env_cfgs[self.policy_idx[i]]["gp"][0]] * 4 for i in range(self.num_envs)], device=self.device, dtype=gs.tc_float)

        # self.x_offset = torch.tensor([self.cpg_env_cfgs[self.policy_idx[i]]["x_offset"] for i in range(self.num_envs)], device=self.device, dtype=gs.tc_float)
        # self.y_offset = torch.tensor([self.cpg_env_cfgs[self.policy_idx[i]]["y_offset"] for i in range(self.num_envs)], device=self.device, dtype=gs.tc_float)
        # self.h_offset = torch.tensor([self.cpg_env_cfgs[self.policy_idx[i]]["h_offset"] for i in range(self.num_envs)], device=self.device, dtype=gs.tc_float)

        # Residual
        self.res = torch.zeros((self.num_envs, 12), device=self.device, dtype=gs.tc_float)
        self.last_res = torch.zeros((self.num_envs, 12), device=self.device, dtype=gs.tc_float)
        self.d_res = torch.zeros((self.num_envs, 12), device=self.device, dtype=gs.tc_float)

        self.res_scale = torch.tensor(self.env_cfg["res_scale"], device=self.device, dtype=gs.tc_float).repeat(self.num_envs, 1)
        self.d_res_scale = torch.tensor(self.env_cfg["d_res_scale"], device=self.device, dtype=gs.tc_float).repeat(self.num_envs, 1)
        

        self.terrain.geoms[0].set_friction(self.env_cfg["friction"][0])

    def get_idx_by_name(self, names):
        indices = []
        for name in names:
            for i in range(len(self.robot.links)):
                if self.robot.links[i].name == name:
                    indices.append(self.robot.links[i].idx)
                    break
        return indices

    def _resample_commands(self, envs_idx):
        # policy_idxからenvs_idxに対応するインデックスを取得
        policy_indices = self.policy_idx[envs_idx]

        # 各インデックスに対応するlin_vel_x_rangeを取得
        lin_vel_x_ranges = torch.tensor(
            [self.cpg_commands_cfgs[idx]["lin_vel_x_range"] for idx in policy_indices],
            device=self.device
        )
        lin_vel_y_ranges = torch.tensor(
            [self.cpg_commands_cfgs[idx]["lin_vel_y_range"] for idx in policy_indices],
            device=self.device
        )
        ang_vel_ranges = torch.tensor(
            [self.cpg_commands_cfgs[idx]["ang_vel_range"] for idx in policy_indices],
            device=self.device
        )
        # 各範囲に対してランダムな値を生成
        random_values_lin_vel_x = gs_rand_float(
            lin_vel_x_ranges[:, 0], lin_vel_x_ranges[:, 1], (len(envs_idx),), self.device
        )
        random_values_lin_vel_y = gs_rand_float(
            lin_vel_y_ranges[:, 0], lin_vel_y_ranges[:, 1], (len(envs_idx),), self.device
        )
        random_values_ang_vel = gs_rand_float(
            ang_vel_ranges[:, 0], ang_vel_ranges[:, 1], (len(envs_idx),), self.device
        )
        # commandsを更新
        self.commands[envs_idx, 0] = random_values_lin_vel_x
        self.commands[envs_idx, 1] = random_values_lin_vel_y
        self.commands[envs_idx, 2] = random_values_ang_vel
    
    def _reset_pd_params(self):
        random_factors_kp = np.random.uniform(-1, 1)
        random_factors_kd = np.random.uniform(-1, 1)
        if self.eval:
            random_factors_kp = 0
            random_factors_kd = 0
        self.kp = np.array(self.env_cfg["kp"]) + np.array(self.env_cfg["kp_random_scale"]) * random_factors_kp
        self.kd = np.array(self.env_cfg["kd"]) + np.array(self.env_cfg["kd_random_scale"]) * random_factors_kd
        self.robot.set_dofs_kp(self.kp.tolist(), self.motor_dofs)
        self.robot.set_dofs_kv(self.kd.tolist(), self.motor_dofs)

    def CPG(self, actions):

        
        self.mu = (self.action_scale_mu[:,:,1] - self.action_scale_mu[:,:,0])/2 * (actions[:,0:4]) + (self.action_scale_mu[:,:,0]+self.action_scale_mu[:,:,1])/2
        self.omega = (self.action_scale_omega[:,:,1] - self.action_scale_omega[:,:,0])/2 * (actions[:,4:8]) + (self.action_scale_omega[:,:,0]+self.action_scale_omega[:,:,1])/2
        self.psi = (self.action_scale_psi[:,:,1] - self.action_scale_psi[:,:,0])/2 * (actions[:,8:12]) + (self.action_scale_psi[:,:,0]+self.action_scale_psi[:,:,1])/2


        self.mu = torch.clip(self.mu, self.action_scale_mu[:,:,0], self.action_scale_mu[:,:,1])
        self.omega = torch.clip(self.omega, self.action_scale_omega[:,:,0], self.action_scale_omega[:,:,1])
        self.psi = torch.clip(self.psi, self.action_scale_psi[:,:,0], self.action_scale_psi[:,:,1])
    

        self.r_ddot = self.a*(self.a*(self.mu - self.r)/4 - self.r_dot)*self.dt
        self.r_dot = self.r_dot + self.r_ddot*self.dt
        self.r = self.r + self.r_dot*self.dt
        self.theta = self.theta + 2*torch.pi*self.omega*self.dt
        self.phi = self.phi + 2*torch.pi*self.psi*self.dt
        self.theta = self.theta % (2*torch.pi)
        self.phi = self.phi % (2*torch.pi)

    def Trajectory(self):


        # Determine g based on the sign of sin(theta)
        g = torch.where(torch.sin(self.theta) > 0, self.gc, self.gp)

        # Calculate x, y, and z positions
        x = -self.d * (self.r-1) * torch.cos(self.theta) * torch.cos(self.phi)
        y = -self.d * (self.r-1) * torch.cos(self.theta) * torch.sin(self.phi)
        # A1
        yr = -0.0838
        yl = 0.0838
        # Go2
        # yr = -0.094
        # yl = 0.094

        if self.env_cfg.get("eth_z", False):

            k = 2 * self.theta / np.pi

            # 条件に基づいてzを計算
            condition1 = (0 <= self.theta) & (self.theta <= np.pi / 2)
            condition2 = (np.pi / 2 < self.theta) & (self.theta <= np.pi)

            z = torch.where(
                condition1,
                -self.h + self.gc * (-2 * k**3 + 3 * k**2),
                torch.where(
                    condition2,
                    -self.h + self.gc * (2 * (k - 1)**3 - 3 * (k - 1)**2 + 1),
                    -self.h
                )
            )

        else:
            z = -self.h + g * torch.sin(self.theta)

        # Construct the trajectory positions
        # foot_pos = torch.stack([
        #     torch.stack([x[:, 0]+self.x_offset[:,0], y[:, 0] + yr + self.y_offset[:,0], z[:, 0]+self.h_offset[:,0]], dim=-1),
        #     torch.stack([x[:, 1]+self.x_offset[:,1], y[:, 1] + yl + self.y_offset[:,1], z[:, 1]+self.h_offset[:,1]], dim=-1),
        #     torch.stack([x[:, 2]+self.x_offset[:,2], y[:, 2] + yr + self.y_offset[:,2], z[:, 2]+self.h_offset[:,2]], dim=-1),
        #     torch.stack([x[:, 3]+self.x_offset[:,3], y[:, 3] + yl + self.y_offset[:,3], z[:, 3]+self.h_offset[:,3]], dim=-1)
        # ], dim=1)
        foot_pos = torch.stack([
            torch.stack([x[:, 0]+self.env_cfg["x_offset"][0], y[:, 0] + yr + self.env_cfg["y_offset"][0], z[:, 0]+self.env_cfg["h_offset"][0]], dim=-1),
            torch.stack([x[:, 1]+self.env_cfg["x_offset"][1], y[:, 1] + yl + self.env_cfg["y_offset"][1], z[:, 1]+self.env_cfg["h_offset"][1]], dim=-1),
            torch.stack([x[:, 2]+self.env_cfg["x_offset"][2], y[:, 2] + yr + self.env_cfg["y_offset"][2], z[:, 2]+self.env_cfg["h_offset"][2]], dim=-1),
            torch.stack([x[:, 3]+self.env_cfg["x_offset"][3], y[:, 3] + yl + self.env_cfg["y_offset"][3], z[:, 3]+self.env_cfg["h_offset"][3]], dim=-1)
        ], dim=1)

        return foot_pos 
    
    import torch

    def Inverse_Kinematics(self, target_positions: torch.Tensor) -> torch.Tensor:
        
        # A1
        L1 = 0.0838
        L2 = 0.2
        L3 = 0.2
        # Go2
        # L1 = 0.094
        # L2 = 0.21
        # L3 = 0.21

        # Initialize a tensor to store joint angles
        joint_angles = torch.zeros(target_positions.shape[:-1] + (3,), device=target_positions.device)
        
        # Calculate th1 for right and left legs
        th_f_yz_right = torch.atan2(-target_positions[:, 0::2, 2], -target_positions[:, 0::2, 1])
        th1_right = th_f_yz_right - torch.acos(L1 / torch.sqrt(target_positions[:, 0::2, 1]**2 + target_positions[:, 0::2, 2]**2))
        
        th_f_yz_left = torch.atan2(target_positions[:, 1::2, 2], target_positions[:, 1::2, 1])
        th1_left = th_f_yz_left + torch.acos(L1 / torch.sqrt(target_positions[:, 1::2, 1]**2 + target_positions[:, 1::2, 2]**2))
        
        # Assign th1 to joint_angles
        joint_angles[:, 0::2, 0] = th1_right
        joint_angles[:, 1::2, 0] = th1_left
        
        # Calculate rotated target positions
        cos_th1 = torch.cos(joint_angles[..., 0])
        sin_th1 = torch.sin(joint_angles[..., 0])
        
        rotated_target_pos = torch.stack([
            target_positions[..., 0],
            cos_th1 * target_positions[..., 1] + sin_th1 * target_positions[..., 2],
            -sin_th1 * target_positions[..., 1] + cos_th1 * target_positions[..., 2]
        ], dim=-1)
        
        # Calculate phi
        phi = torch.acos((L2**2 + L3**2 - rotated_target_pos[..., 0]**2 - rotated_target_pos[..., 2]**2) / (2 * L2 * L3))
        
        # Calculate th2 and th3
        th_f_xz = torch.atan2(-rotated_target_pos[..., 0], -rotated_target_pos[..., 2])
        th2 = th_f_xz + (torch.pi - phi) / 2
        th3 = -torch.pi + phi
        
        # Assign th2 and th3 to joint_angles
        joint_angles[..., 1] = th2
        joint_angles[..., 2] = th3
        
        return joint_angles.view(target_positions.shape[0], -1)
    
    # def dof_limit(self,target_dof_pos):
    #     return torch.clip(target_dof_pos, -self.env_cfg["dq_limit"] * self.dt + self.last_target_dof_pos, self.env_cfg["dq_limit"] * self.dt + self.last_target_dof_pos)


    def step(self, actions):

        self.res = actions * self.d_res_scale
        self.res = torch.clip(self.res, -self.d_res_scale*self.dt + self.last_res, self.d_res_scale*self.dt + self.last_res)
        self.res = torch.clip(self.res, -self.res_scale, self.res_scale)
        self.d_res = (self.res - self.last_res) / self.dt
        self.last_res = self.res

        # self.d_res = actions * self.d_res_scale
        # self.res = self.res + self.d_res * self.dt
        # self.res = torch.clip(self.res, -self.res_scale, self.res_scale)

        cpg_actions_list = []
        for i in range(len(self.cpg_policy)):
            _cpg_actions = self.cpg_policy[i](self.obs_buf_cpg)
            _cpg_actions = torch.clip(_cpg_actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
            cpg_actions_list.append(_cpg_actions)
        
        # self.cpg_actions = cpg_actions_list[torch.arange(len(self.cpg_actions)), self.policy_idx]

        self.cpg_actions = torch.stack([cpg_actions_list[self.policy_idx[i]][i] for i in range(self.num_envs)], dim=0)
        
        
        cpg_exec_actions = self.last_cpg_actions if self.simulate_action_latency else self.cpg_actions
        if self.env_cfg["pd_stop"]: 
            cpg_exec_actions[self.pd_stop_idx] = torch.tensor([-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,0.0,0.0,0.0,0.0], device=self.device)

        self.CPG(cpg_exec_actions)
        foot_pos = self.Trajectory()
        target_dof_pos = self.Inverse_Kinematics(foot_pos)
        # target_dof_pos = self.dof_limit(target_dof_pos)
        # self.last_target_dof_pos = target_dof_pos
        target_dof_pos = target_dof_pos + self.res
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.dof_force[:] = self.robot.get_dofs_force(self.motor_dofs)



        forces = self.robot.get_links_net_contact_force()
        self.foot_forces = torch.stack([forces[:, idx - 1] for idx in self.calf_link_idx], dim=1)
        self.foot_force_norms = torch.norm(self.foot_forces, dim=2)
        self.foot_contact = (self.foot_force_norms >= 1).int()


        # self.foot_force_FR = self.robot.get_links_net_contact_force()[:,self.calf_link_idx[0]-1]
        # self.foot_force_FL = self.robot.get_links_net_contact_force()[:,self.calf_link_idx[1]-1]
        # self.foot_force_RR = self.robot.get_links_net_contact_force()[:,self.calf_link_idx[2]-1]
        # self.foot_force_RL = self.robot.get_links_net_contact_force()[:,self.calf_link_idx[3]-1]
        # print(self.foot_force_FR)
        # print(self.foot_force_FL)
        # print(self.foot_force_RR)
        # print(self.foot_force_RL)

        # print(self.foot_force_FR[:,0]+self.foot_force_FR[:,1]+self.foot_force_FR[:,2])
        # print(self.foot_force_FL[:,0]+self.foot_force_FL[:,1]+self.foot_force_FL[:,2])
        # print(self.foot_force_RR[:,0]+self.foot_force_RR[:,1]+self.foot_force_RR[:,2])
        # print(self.foot_force_RL[:,0]+self.foot_force_RL[:,1]+self.foot_force_RL[:,2])
        # print(self.foot_force)
        # print(self.robot.links[0].name)
        # print(self.robot.links[0].idx)
        # print(len(self.robot.links))

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        if len(envs_idx) > 0:
            self._resample_commands(envs_idx)

        if self.last_gain_reset > self.env_cfg["gain_resampling_time_s"] and not self.eval:
            self._reset_pd_params()
            self.last_gain_reset = 0
        else:
            self.last_gain_reset += self.dt
        
        if self.last_friction_reset > self.env_cfg["friction_resampling_time_s"] and not self.eval:
            friction = (self.env_cfg["friction"][2]-self.env_cfg["friction"][1])*np.random.uniform(0, 1) + self.env_cfg["friction"][1]
            self.terrain.geoms[0].set_friction(friction)
            self.last_friction_reset = 0
        else:
            self.last_friction_reset += self.dt

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew


        

        noise_dof_pos = torch.normal(mean=0.0, std=self.noise_std_dof_pos, size=(self.num_envs, 12), device=self.device)
        noise_dof_vel = torch.normal(mean=0.0, std=self.noise_std_dof_vel, size=(self.num_envs, 12), device=self.device)
        noise_euler = torch.normal(mean=0.0, std=self.noise_std_euler, size=(self.num_envs, 2), device=self.device)
        noise_ang_vel = torch.normal(mean=0.0, std=self.noise_std_ang_vel, size=(self.num_envs, 3), device=self.device)

        # compute observations
        noise_dof_pos = torch.normal(mean=0.0, std=self.noise_std_dof_pos, size=(self.num_envs, 12), device=self.device)
        noise_dof_vel = torch.normal(mean=0.0, std=self.noise_std_dof_vel, size=(self.num_envs, 12), device=self.device)
        noise_euler = torch.normal(mean=0.0, std=self.noise_std_euler, size=(self.num_envs, 2), device=self.device)
        noise_ang_vel = torch.normal(mean=0.0, std=self.noise_std_ang_vel, size=(self.num_envs, 3), device=self.device)

        # Observations
        obs_components = []
        cpg_obs_components = []
        if self.obs_cfg.get("include_dof_pos", False):
            obs_components.append((self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"] + noise_dof_pos)  # 12

        if self.cpg_obs_cfgs[0].get("include_dof_pos", False):
            cpg_obs_components.append((self.dof_pos - self.default_dof_pos) * self.cpg_obs_scales["dof_pos"] + noise_dof_pos)  # 12

        if self.obs_cfg.get("include_dof_vel", False):
            obs_components.append(self.dof_vel * self.obs_scales["dof_vel"] + noise_dof_vel)  # 12

        if self.cpg_obs_cfgs[0].get("include_dof_vel", False):
            cpg_obs_components.append(self.dof_vel * self.cpg_obs_scales["dof_vel"] + noise_dof_vel)  # 12

        if self.obs_cfg.get("include_euler", False):
            obs_components.append(self.base_euler[:, 0:2] * self.obs_scales["euler"] + noise_euler)  # 2

        if self.cpg_obs_cfgs[0].get("include_euler", False):
            cpg_obs_components.append(self.base_euler[:, 0:2] * self.cpg_obs_scales["euler"] + noise_euler)  # 2

        if self.obs_cfg.get("include_ang_vel", False):
            obs_components.append(self.base_ang_vel * self.obs_scales["ang_vel"] + noise_ang_vel)  # 3

        if self.cpg_obs_cfgs[0].get("include_ang_vel", False):
            cpg_obs_components.append(self.base_ang_vel * self.cpg_obs_scales["ang_vel"] + noise_ang_vel)  # 3

        if self.obs_cfg.get("include_foot_contact", False):
            obs_components.append(self.foot_contact * self.obs_scales["foot_contact"])  # 4

        if self.cpg_obs_cfgs[0].get("include_foot_contact", False):
            cpg_obs_components.append(self.foot_contact * self.cpg_obs_scales["foot_contact"])  # 4

        # CPG state
        if self.obs_cfg.get("include_mu", False):
            obs_components.extend([
                self.mu * self.obs_scales["mu"],  # 4
            ])
        if self.cpg_obs_cfgs[0].get("include_mu", False):
            cpg_obs_components.extend([
                self.mu * self.cpg_obs_scales["mu"],  # 4
            ])
        if self.obs_cfg.get("include_omega", False):
            obs_components.extend([
                self.omega * self.obs_scales["omega"],  # 4
            ])
        if self.cpg_obs_cfgs[0].get("include_omega", False):
            cpg_obs_components.extend([
                self.omega * self.cpg_obs_scales["omega"],  # 4
            ])
        if self.obs_cfg.get("include_psi", False):
            obs_components.extend([
                self.psi * self.obs_scales["psi"],  # 4
            ])
        if self.cpg_obs_cfgs[0].get("include_psi", False):
            cpg_obs_components.extend([
                self.psi * self.cpg_obs_scales["psi"],  # 4
            ])
        if self.obs_cfg.get("include_r", False):
            obs_components.extend([
                self.r * self.obs_scales["r"],  # 4
            ])
        if self.cpg_obs_cfgs[0].get("include_r", False):
            cpg_obs_components.extend([
                self.r * self.cpg_obs_scales["r"],  # 4
            ])
        if self.obs_cfg.get("include_theta", False):
            obs_components.extend([
                self.theta * self.obs_scales["theta"],  # 4
            ])
        if self.cpg_obs_cfgs[0].get("include_theta", False):
            cpg_obs_components.extend([
                self.theta * self.cpg_obs_scales["theta"],  # 4
            ])
        if self.obs_cfg.get("include_phi", False):
            obs_components.extend([
                self.phi * self.obs_scales["phi"],  # 4
            ])
        if self.cpg_obs_cfgs[0].get("include_phi", False):
            cpg_obs_components.extend([
                self.phi * self.cpg_obs_scales["phi"],  # 4
            ])

        # command
        if self.obs_cfg.get("include_commands", False):
            obs_components.append(self.commands * self.commands_scale)  # 3

        if self.cpg_obs_cfgs[0].get("include_commands", False):
            cpg_obs_components.append(self.commands * self.cpg_commands_scale)  # 3

        self.obs_buf_cpg = torch.cat(cpg_obs_components, axis=-1)

        if self.obs_cfg.get("include_res", False):
            obs_components.append(self.res * self.obs_scales["res"])  # 12
        if self.obs_cfg.get("include_d_res", False):
            obs_components.append(self.d_res * self.obs_scales["d_res"])  # 12

        self.obs_buf = torch.cat(obs_components, axis=-1)


        self.last_actions[:] = self.actions[:]
        self.last_cpg_actions[:] = self.cpg_actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        if self.capture:
            # self.cam.set_pose(
            #     pos=(self.base_pos[0, 0].cpu().numpy() + 1.0, self.base_pos[0, 1].cpu().numpy() + 1.0, self.base_pos[0, 2].cpu().numpy() + 0.5),
            # lookat=(self.base_pos[0, 0].cpu().numpy(), self.base_pos[0, 1].cpu().numpy(), self.base_pos[0, 2].cpu().numpy()),
            #     up=(0.0, 0.0, 1.0),
            # )
            self.cam.set_pose(
                pos=(self.base_pos[0, 0].cpu().numpy() + 0.0, self.base_pos[0, 1].cpu().numpy() - 8.0, self.base_pos[0, 2].cpu().numpy() + 2.0),
            lookat=(self.base_pos[0, 0].cpu().numpy(), self.base_pos[0, 1].cpu().numpy(), self.base_pos[0, 2].cpu().numpy()),
                up=(0.0, 0.0, 1.0),
            )
            self.cam.render()
        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        
        self.mu[envs_idx] = torch.ones(4,device=self.device) # デフォルト
        self.omega[envs_idx] = torch.zeros(4,device=self.device) # デフォルト
        self.psi[envs_idx] = torch.zeros(4,device=self.device) # デフォルト

        # self.mu[envs_idx] = torch.rand(4,device=self.device)*(self.env_cfg["action_scale_mu"][1]-self.env_cfg["action_scale_mu"][0])+self.env_cfg["action_scale_mu"][0] # デフォルト消す
        # self.omega[envs_idx] = torch.rand(4,device=self.device)*(self.env_cfg["action_scale_omega"][1]-self.env_cfg["action_scale_omega"][0])+self.env_cfg["action_scale_omega"][0] # デフォルト消す
        # self.psi[envs_idx] = torch.rand(4,device=self.device)*(self.env_cfg["action_scale_psi"][1]-self.env_cfg["action_scale_psi"][0])+self.env_cfg["action_scale_psi"][0] # デフォルト消す



        self.r[envs_idx] = 1 + torch.rand(4,device=self.device)
        if self.eval:
            self.r[envs_idx] = torch.ones(4,device=self.device)
        # self.r[envs_idx] = torch.ones(4,device=self.device)

        # self.theta[envs_idx] = 2*torch.pi*torch.rand(4,device=self.device) # デフォルト消す
        

        # self.phi[envs_idx] = torch.zeros(4,device=self.device) # デフォルト
        self.phi[envs_idx] = (torch.pi*torch.rand(4,device=self.device) - torch.pi/2) % (2*torch.pi) # デフォルト消す
        if self.env_cfg["pd_stop"]:
            self.phi[self.pd_stop_idx] = torch.zeros(4,device=self.device)





        # reset dofs
        foot_pos = self.Trajectory()
        target_dof_pos = self.Inverse_Kinematics(foot_pos)
        self.dof_pos[envs_idx] = target_dof_pos[envs_idx]
        # self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos

        self.base_pos[envs_idx, 0] = (2*torch.rand(len(envs_idx),device=self.device)-1) * self.env_cfg["init_pos_range"][0]
        self.base_pos[envs_idx, 1] = (2*torch.rand(len(envs_idx),device=self.device)-1) * self.env_cfg["init_pos_range"][1]

        # self.base_pos[envs_idx, 0] = torch.randint(0, self.env_cfg["n_subterrains"][0], (len(envs_idx),), device=self.device) * self.env_cfg["subterrain_size"][0] + self.env_cfg["subterrain_size"][0]/2 + (2*torch.rand(len(envs_idx),device=self.device)-1) * 3
        # self.base_pos[envs_idx, 1] = torch.randint(0, self.env_cfg["n_subterrains"][1], (len(envs_idx),), device=self.device) * self.env_cfg["subterrain_size"][1] + self.env_cfg["subterrain_size"][1]/2 + (2*torch.rand(len(envs_idx),device=self.device)-1) * 3
        # self.base_pos[envs_idx, 0] = torch.randint(0, self.env_cfg["n_subterrains"][0], (len(envs_idx),), device=self.device) * self.env_cfg["subterrain_size"][0] + self.env_cfg["subterrain_size"][0]/2 
        # self.base_pos[envs_idx, 1] = torch.randint(0, self.env_cfg["n_subterrains"][1], (len(envs_idx),), device=self.device) * self.env_cfg["subterrain_size"][1] + self.env_cfg["subterrain_size"][1]/2 

        # x_idx = (self.base_pos[envs_idx, 0] // self.env_cfg["horizontal_scale"]).long().cpu()
        # y_idx = (self.base_pos[envs_idx, 1] // self.env_cfg["horizontal_scale"]).long().cpu()
        # height_values = self.height_field[x_idx, y_idx]  # NumPy配列として取得
        # height_values_tensor = torch.tensor(height_values, device=self.device, dtype=gs.tc_float)  # NumPy配列をPyTorchテンソルに変換
        # self.base_pos[envs_idx, 2] = self.env_cfg["vertical_scale"] * height_values_tensor + self.env_cfg["h"][0] + 0.05

        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_cpg_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

        if not self.eval:
            random_values_h = torch.rand(len(envs_idx), 1, device=self.device)  # Generate one random value per environment
            self.h[envs_idx] = (self.env_cfg["h"][2] - self.env_cfg["h"][1]) * random_values_h + self.env_cfg["h"][1]
            random_values_gc = torch.rand(len(envs_idx), 1, device=self.device)  # Generate one random value per environment
            self.gc[envs_idx] = (self.env_cfg["gc"][2] - self.env_cfg["gc"][1]) * random_values_gc + self.env_cfg["gc"][1]
            random_values_gp = torch.rand(len(envs_idx), 1, device=self.device)  # Generate one random value per environment
            self.gp[envs_idx] = (self.env_cfg["gp"][2] - self.env_cfg["gp"][1]) * random_values_gp + self.env_cfg["gp"][1]
        else:
            self.h[envs_idx] = self.env_cfg["h"][0]
            self.gc[envs_idx] = self.env_cfg["gc"][0]
            self.gp[envs_idx] = self.env_cfg["gp"][0]

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])
    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_x(self):
        # Penalize xy axis base angular velocity
        return torch.square(self.base_ang_vel[:, 0])
    
    def _reward_ang_vel_y(self):
        # Penalize xy axis base angular velocity
        return torch.square(self.base_ang_vel[:, 1])

    def _reward_work(self):
        # return torch.sum(torch.abs(self.dof_force * (self.dof_vel - self.last_dof_vel)), dim=1)
        return torch.sum(torch.abs(self.dof_force * self.dof_vel), dim=1)

    def _reward_foot_force_norm(self):
        self.foot_force_norms_ref = torch.ones((self.num_envs,4),device=self.device)
        foot_force_norm_error = torch.tanh((self.foot_force_norms - self.reward_cfg["foot_force_norms_threshold"][0])*(2/(self.reward_cfg["foot_force_norms_threshold"][1]-self.reward_cfg["foot_force_norms_threshold"][0])))
        foot_force_norm_error = torch.clip(foot_force_norm_error, 0.0, 1.0)
        return torch.sum(foot_force_norm_error, dim=1)
    
    # def _reward_body_height(self):
    #     x_idx = (self.base_pos[:, 0] // self.env_cfg["horizontal_scale"]).long().cpu()
    #     y_idx = (self.base_pos[:, 1] // self.env_cfg["horizontal_scale"]).long().cpu()
    #     height_values = self.height_field[x_idx, y_idx]  # NumPy配列として取得
    #     height_values_tensor = torch.tensor(height_values, device=self.device, dtype=gs.tc_float)  # NumPy配列をPyTorchテンソルに変換
    #     return torch.abs(self.base_pos[:, 2] - height_values_tensor - self.env_cfg["h"][0])

    def _reward_res(self):
        return torch.sum(torch.abs(self.res),dim=1)
    
    def _reward_d_res(self):
        return torch.sum(torch.abs(self.d_res),dim=1)
    
    def _reward_euler_roll(self):
        return torch.abs(self.base_euler[:, 0])
    
    def _reward_euler_pitch(self):
        return torch.abs(self.base_euler[:, 1])
    
    def put_box(self,n_row=10,n_col=10,box_size=0.3,box_height_range=[0.01,0.12]):
        self.boxes = []
        colors = ["#fafcfc","#74878f",]
        cmap = mcolors.LinearSegmentedColormap.from_list("custom",colors)
        for i in range(2*n_row+1):
            x = box_size*(i-(n_row+1))
            for j in range(2*n_col+1):
                if (i+j)%2 == 0:
                    y = box_size*(j-(n_col+1))
                    box_height = (box_height_range[1] - box_height_range[0]) * np.random.rand() + box_height_range[0]

                    rgba = self.value_to_rgba(box_height/box_height_range[1],cmap,box_height_range[0],box_height_range[1])
                    box = self.scene.add_entity(
                        gs.morphs.Box(
                            size=(box_size,box_size,box_height),
                            pos=(x,y,box_height/2),
                            quat=(0,0,0,1),
                            visualization=True,
                            collision=True,
                            fixed=True,
                        ),
                        
                        surface=gs.surfaces.Default(
                            color=rgba,
                        ),
                    )
                    friction = (self.env_cfg["friction"][2] - self.env_cfg["friction"][1]) * np.random.rand() + self.env_cfg["friction"][1]
                    box.geoms[0].set_friction(friction)
                    self.boxes.append(box)

    def value_to_rgba(self,value,cmap=plt.get_cmap("viridis"),vmin=0.0,vmax=1.0):
        value = np.clip(value, vmin, vmax)
        value = (value - vmin) / (vmax - vmin)
        rgba = cmap(value)
        return rgba
    
class A1CPGEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, device="cuda:0",eval=True):
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.eval = eval
  
        self.obs_buf = torch.zeros(num_envs, self.num_obs, device=self.device, dtype=gs.tc_float)

    def reset(self):
        return self.obs_buf, None

if __name__ == "__main__":
    def get_cfgs():
        env_cfg = {
            "num_actions": 12,
            "logs": ["250216_161050","250215_215635","250216_115210","250216_111642","250216_131807"],
            # joint/link names
            "default_joint_angles": {  # [rad]
                "FL_hip_joint": 0.0,
                "FR_hip_joint": 0.0,
                "RL_hip_joint": 0.0,
                "RR_hip_joint": 0.0,
                "FL_thigh_joint": 1.0,
                "FR_thigh_joint": 1.0,
                "RL_thigh_joint": 1.0,
                "RR_thigh_joint": 1.0,
                "FL_calf_joint": -1.5,
                "FR_calf_joint": -1.5,
                "RL_calf_joint": -1.5,
                "RR_calf_joint": -1.5,
            },
            "dof_names": [
                "FR_hip_joint",
                "FR_thigh_joint",
                "FR_calf_joint",
                "FL_hip_joint",
                "FL_thigh_joint",
                "FL_calf_joint",
                "RR_hip_joint",
                "RR_thigh_joint",
                "RR_calf_joint",
                "RL_hip_joint",
                "RL_thigh_joint",
                "RL_calf_joint",
            ],
            # PD
            "kp": [65.0,65.0,65.0,65.0,65.0,65.0,65.0,65.0,65.0,65.0,65.0,65.0],
            # "kp": [80.0,80.0,80.0,80.0,80.0,80.0,80.0,80.0,80.0,80.0,80.0,80.0],
            # "kp": [100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0],
            "kd": [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
            # "kp_random_scale": [5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0],
            # "kd_random_scale": [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5],
            "kp_random_scale": [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
            "kd_random_scale": [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
            # termination
            # "termination_if_roll_greater_than": 30,  # degree
            # "termination_if_pitch_greater_than": 30,
            "termination_if_roll_greater_than": 180,  # degree
            "termination_if_pitch_greater_than": 180,
            # base pose
            "base_init_pos": [18.0, 30.0, 0.38],
            "base_init_quat": [1.0, 0.0, 0.0, 0.0],
            "episode_length_s": 20.0,
            "resampling_time_s": 4.0,
            "gain_resampling_time_s": 2.0,
            "friction_resampling_time_s": 2.0,
            # "action_scale_mu": [1.0, 2.0],
            # "action_scale_omega": [0.0, 2.0],
            # # "action_scale_omega": [-1.5, 1.5],
            # "action_scale_psi": [-1.5, 1.5],
            # # "action_scale_psi": [-0.5, 0.5],
            "pd_stop":True,
            "pd_stop_interval":10,
            # "n_subterrains":(2,2),
            # "n_subterrains":(1,1),
            # # "subterrain_types":[["flat_terrain","fractal_terrain"],["fractal_terrain","flat_terrain"]],
            # # "subterrain_types":[["flat_terrain","fractal_terrain"]],
            # "subterrain_types":[["stairs_terrain"]],
            # "subterrain_size":(20,20),
            # "horizontal_scale":0.25,
            # "vertical_scale":0.02,
            "n_row":20,
            "n_col":20,
            "box_size":0.3,
            "box_height_range":[0.01,0.10],
            "init_pos_range":[5,5],

            "res_scale":[0.5,1.0,0.5,0.5,1.0,0.5,0.5,1.0,0.5,0.5,1.0,0.5], # rad
            "d_res_scale":[5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0], # rad/s

            "a":150,
            "d":0.10,
            "h":[0.27,0.27,0.27],
            "h_offset":[0.0,0.0,-0.0,-0.0],
            "x_offset":[0.0,0.0,-0.0,-0.0],
            "y_offset":[0.0,0.0,0.0,0.0],
            "gc":[0.20,0.20,0.20],
            "gp":[0.00,0.00,0.00],
            "friction":[2.0,2.0,2.0],
            "simulate_action_latency": True,
            "clip_actions": 1.0,
            "eth_z": True,
            }
        # obs_cfg = {
        #         "num_obs": 56,
        #         "obs_scales": {
        #             "lin_vel": 2.0,
        #             "dof_pos": 1.0,
        #             "dof_vel": 0.05,
        #             "ang_vel": 0.25,
        #             "euler": 0.5,
        #             "mu": 1.0,
        #             "omega": 0.3,
        #             "psi": 0.8,
        #             "r": 1.0,
        #             "theta": 0.3,
        #             "phi": 0.3,
        #         },
        #     }
        obs_cfg = {
                "num_obs": 84,
                "obs_scales": {
                    "lin_vel": 1.0,
                    "dof_pos": 1.0,
                    "dof_vel": 1.0,
                    "ang_vel": 1.0,
                    "euler": 1.0,
                    "foot_contact": 1.0,
                    "mu": 1.0,
                    "omega": 1.0,
                    "psi": 1.0,
                    "r": 1.0,
                    "theta": 1.0,
                    "phi": 1.0,
                    "res": 1.0,
                    "d_res": 1.0,
                },
                "noise_std_dof_pos": 0.00,
                "noise_std_dof_vel": 0.00,
                "noise_std_euler": 0.00,
                "noise_std_ang_vel": 0.00,
                "include_dof_pos": True,
                "include_dof_vel": True,
                "include_euler": True,
                "include_ang_vel": True,
                "include_foot_contact": True,
                "include_mu": True,
                "include_omega": True,
                "include_psi": True,
                "include_r": True,
                "include_theta": True,
                "include_phi": True,
                "include_commands": True,
                "include_res": True,
                "include_d_res": True,
            }
        # obs_cfg = {
        #     "num_obs": 45,
        #     "obs_scales": {
        #         "lin_vel": 2.0,
        #         "ang_vel": 0.25,
        #         "dof_pos": 1.0,
        #         "dof_vel": 0.05,
        #     },
        # }
        reward_cfg = {
            "tracking_sigma": 0.25,
            # "tracking_sigma": 0.1,
            "foot_force_norms_threshold": [80, 200],
            "reward_scales": {
                "tracking_lin_vel_x": 0.75,
                "tracking_lin_vel_y": 0.75,
                # "tracking_ang_vel": 0.5,
                # "tracking_lin_vel_x": 3.0,
                # "tracking_lin_vel_y": 0.75,
                "tracking_ang_vel": 0.5,
                "lin_vel_z": -2.0,
                "ang_vel_x": -0.05,
                "ang_vel_y": -0.05,
                "work": -0.001,
                # "foot_force_norm": -0.1,
                "res": -0.01,
                "d_res": -0.01,
            },
        }
        command_cfg = {
            "num_commands": 3,
            "lin_vel_x_range": [0.3,0.3],
            # "lin_vel_x_range": [1.0,1.0],
            "lin_vel_y_range": [0.0, 0.0],
            "ang_vel_range": [0.0, 0.0],
        }


        return env_cfg, obs_cfg, reward_cfg, command_cfg

    
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    gs.init()
    
    env = A1Res4CPGEnv(num_envs=2, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg, device="cuda:0",capture=True, eval=True)

    obs, _ = env.reset()
    # env.r = torch.ones(env.num_envs,4,device="cuda:0")
    with torch.no_grad():
        # while True:
        if env.capture:
            env.cam.start_recording()
        for i in range(10):
            actions = torch.zeros(env.num_envs,env.num_actions,device="cuda:0")
            # actions[0,0:4] = -100
            # actions[0,4:8] = -100
            obs, _, rews, dones, infos = env.step(actions)
            # print(env.r,env.theta,env.phi,env.mu,env.omega,env.psi)
            # print(env.base_pos,env.base_euler)
            # print(env.dof_pos,env.r,env.theta,env.phi,env.mu,env.omega,env.psi)
        if env.capture:
            env.cam.stop_recording(save_to_filename=f"{script_dir}/env_video.mp4", fps=60)



# python /mnt/ssd1/ryosei/master/DRL_Genesis/Scripts/Res4CPG-RL/Train.py --device 0 --seed 0

import argparse
import os
import shutil
import json
import torch
from Env import A1Res4CPGEnv
from rsl_rl.runners import OnPolicyRunner

import re
import tqdm
import csv
import genesis as gs
import pytz
import datetime
script_dir = os.path.dirname(os.path.abspath(__file__))

timezone = pytz.timezone('Asia/Tokyo')
start_datetime = datetime.datetime.now(timezone)    
start_formatted = start_datetime.strftime("%y%m%d_%H%M%S")
log_dir = f"{script_dir}/Logs/{start_formatted}"


def get_train_cfg(exp_name, max_iterations, seed=1):

    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            # "entropy_coef": 0.01,
            "entropy_coef": 0.05,
            "gamma": 0.99,
            "lam": 0.95,
            # "lam": 0.99,
            # "learning_rate": 0.001,
            "learning_rate": 0.0005,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            # "actor_hidden_dims": [256, 128, 64],
            # "critic_hidden_dims": [256, 128, 64],
            "init_noise_std": 1.0,
            # "init_noise_std": 0.5,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            # "num_steps_per_env": 24,
            "num_steps_per_env": 96,
            "policy_class_name": "ActorCritic",
            # "policy_class_name": "ActorCriticRecurrent",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 25,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": seed,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # "logs": ["250216_161050","250215_215635","250216_115210","250216_111642","250216_131807"],
        # "logs": ["250219_032206"], # Stop
        # "logs": ["250219_021030"], # Forward
        "logs": ["250219_032206","250219_021030","250219_025005","250219_032033","250219_034230"],
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
        "termination_if_roll_greater_than": 40,  # degree
        "termination_if_pitch_greater_than": 40,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.37],
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
        # "n_subterrains":(1,2),
        # # "subterrain_types":[["flat_terrain","fractal_terrain"],["fractal_terrain","flat_terrain"]],
        # "subterrain_types":[["fractal_terrain","flat_terrain"]],
        # "subterrain_size":(20,20),
        # "horizontal_scale":0.25,
        # "vertical_scale":0.05,
        "n_row":7,
        "n_col":7,
        "box_size":0.5,
        "box_height_range":[0.01,0.12],
        "init_pos_range":[1,1],

        # "res_scale":[0.5,1.0,0.5,0.5,1.0,0.5,0.5,1.0,0.5,0.5,1.0,0.5], # rad
        # "res_scale":[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5], # rad
        "res_scale":[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5], # rad
        # "res_scale":[0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25], # rad
        "d_res_scale":[5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0], # rad/s
        # "d_res_scale":[2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0], # rad/s

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
        # "tracking_sigma": 0.25,
        "tracking_sigma": 0.1,
        "foot_force_norms_threshold": [80, 200],
        "reward_scales": {
            "tracking_lin_vel_x": 0.75,
            "tracking_lin_vel_y": 0.75,
            # "tracking_ang_vel": 0.5,
            # "tracking_lin_vel_x": 3.0,
            # "tracking_lin_vel_y": 0.75,
            "tracking_ang_vel": 0.5,
            "lin_vel_z": -0.05,
            "ang_vel_x": -0.05,
            "ang_vel_y": -0.05,
            "work": -0.001,
            "res": -0.5,
            "d_res": -0.00001,
            # "euler_roll": -0.01,
            # "euler_pitch": -0.01,
            # "body_height": -0.01,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="A1-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("-I", "--max_iterations", type=int, default=200)
    parser.add_argument("-D", "--device", type=int, default=0)
    parser.add_argument("-S", "--seed", type=int, default=1)
    args = parser.parse_args()

    # Device
    visible_device = args.device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"))

    device = "cuda:0"

    gs.init(logging_level="warning")
    
    # timezone = pytz.timezone('Asia/Tokyo')
    # start_datetime = datetime.datetime.now(timezone)    
    # start_formatted = start_datetime.strftime("%y%m%d_%H%M%S")
    
    # log_dir = f"{script_dir}/Logs/{start_formatted}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations, args.seed)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # 複数の設定データをまとめる
    config_data = {
        "train_cfg": train_cfg,
        "env_cfg": env_cfg,
        "obs_cfg": obs_cfg,
        "reward_cfg": reward_cfg,
        "command_cfg": command_cfg
    }

    # JSONファイルに保存
    with open(f'{log_dir}/config.json', 'w') as json_file:
        json.dump(config_data, json_file, indent=4)

    env = A1Res4CPGEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=False, device=device
    )

    

    runner = OnPolicyRunner(env, train_cfg, f"{log_dir}/Networks", device=device)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


# def eval(command_x=[0.0,0.0], command_y=[0.0,0.0], command_ang=[1.0,1.0]):
def eval(command_x=None, command_y=None, command_ang=None,log_dir=log_dir):


    # 日付、時間が最も新しいフォルダを見つける
    # directory = f"{script_dir}/Logs"
    # folders = [os.path.join(directory, folder) for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]
    # log_dir = max(folders, key=lambda folder: os.path.getmtime(folder))


    files = os.listdir(log_dir+"/Networks")


    
    # ファイル名から数字を抽出し、最大値を持つファイルを見つける
    max_number = -1
    for file in files:
        match = re.search(r'model_(\d+)\.pt$', file)
        if match:
            number = int(match.group(1))
            if number > max_number:
                max_number = number
    ckpt = max_number
    print(f"ckpt: {ckpt}")

    timezone = pytz.timezone('Asia/Tokyo')
    start_datetime = datetime.datetime.now(timezone)    
    start_formatted = start_datetime.strftime("%y%m%d_%H%M%S")
    
    test_dir = f"{log_dir}/Test/{start_formatted}"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir, exist_ok=True)

    with open(f"{log_dir}/config.json", 'r') as json_file:
        loaded_config = json.load(json_file)

    # 各設定を個別に取得
    train_cfg = loaded_config.get("train_cfg")
    env_cfg = loaded_config.get("env_cfg")
    # env_cfg["n_subterrains"] = (1,1)
    # # env_cfg["subterrain_types"] = [["fractal_terrain"]]
    # env_cfg["subterrain_types"] = [[env_cfg["subterrain_types"][0][0]]]
    
    obs_cfg = loaded_config.get("obs_cfg")
    reward_cfg = loaded_config.get("reward_cfg")
    command_cfg = loaded_config.get("command_cfg")
    if command_x is not None:
        command_cfg["lin_vel_x_range"] = command_x
    if command_y is not None:
        command_cfg["lin_vel_y_range"] = command_y
    if command_ang is not None:
        command_cfg["ang_vel_range"] = command_ang
    reward_cfg["reward_scales"] = {}

    env = A1Res4CPGEnv(
        num_envs=6,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        capture=True,
        device="cuda:0",
        eval=True
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, "Networks",f"model_{ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    # Open a CSV file to log actions and observations
    csv_file_path = f"{test_dir}/actions_obs.csv"
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header
        csv_writer.writerow(['step', 'action', 'observation', 'foot_force_norms'])

        obs, _ = env.reset()
        with torch.no_grad():
            env.cam.start_recording()
            for i in tqdm.tqdm(range(500)):
                actions = policy(obs)
                obs, _, rews, dones, infos = env.step(actions)

                # アクションと観測をログに記録
                # csv_writer.writerow([i, actions.cpu().numpy().tolist(), obs.cpu().numpy().tolist()])
                csv_writer.writerow([i, actions.cpu().numpy().tolist(), obs.cpu().numpy().tolist(), env.foot_force_norms.cpu().numpy().tolist()])

            env.cam.stop_recording(save_to_filename=f"{test_dir}/video.mp4", fps=50)


if __name__ == "__main__":
    main()
    eval()


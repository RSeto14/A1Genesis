import argparse
import os
import json
import datetime
import pytz
import shutil
import torch
from Env import A1Res4CPGEnv
from rsl_rl.runners import OnPolicyRunner
import re
import tqdm
import genesis as gs
import csv
script_dir = os.path.dirname(os.path.abspath(__file__))


def main(command_x=None, command_y=None, command_ang=None,log_dir="/mnt/ssd1/ryosei/master/DRL_Genesis/Scripts/Res4CPG-RL/Logs/250217_201224"):

    gs.init(logging_level="warning")


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
    # main()
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=7)
    args = parser.parse_args()

    visible_device = args.device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"))

    main(log_dir="/mnt/ssd1/ryosei/master/DRL_Genesis/Scripts/Res4CPG-RL/Logs/250219_011014")
    
"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""

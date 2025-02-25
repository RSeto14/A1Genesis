import argparse
import os
import json
import datetime
import pytz
import shutil
import torch
from Env import A1CPGEnv
from rsl_rl.runners import OnPolicyRunner
import re
import tqdm
import genesis as gs
import csv
script_dir = os.path.dirname(os.path.abspath(__file__))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="a1-walking")
    parser.add_argument("--log_dir", type=str, default="/mnt/ssd1/ryosei/master/DRL_Genesis/Scripts/CPG-RL/Logs/250215_215635")
    parser.add_argument("--ckpt", type=int, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--capture", type=bool, default=True)
    parser.add_argument("--command_x", type=list, default=[0.3, 0.3])
    parser.add_argument("--command_y", type=list, default=[0.0, 0.0])
    parser.add_argument("--command_ang", type=list, default=[0.0, 0.0])
    args = parser.parse_args()

    # Device
    visible_device = args.device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"))

    device = "cuda:0"

    gs.init(logging_level="warning")

    if args.log_dir is None:
        directory = f"{script_dir}/Logs"
        folders = [os.path.join(directory, folder) for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]

        # 日付、時間が最も新しいフォルダを見つける
        log_dir = max(folders, key=lambda folder: os.path.getmtime(folder))
    else:
        log_dir = args.log_dir

    files = os.listdir(log_dir+"/Networks")

    if args.ckpt is None:
    
        # ファイル名から数字を抽出し、最大値を持つファイルを見つける
        max_number = -1
        for file in files:
            match = re.search(r'model_(\d+)\.pt$', file)
            if match:
                number = int(match.group(1))
                if number > max_number:
                    max_number = number
        args.ckpt = max_number
        print(f"ckpt: {args.ckpt}")

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
    obs_cfg = loaded_config.get("obs_cfg")
    # obs_cfg["sensor_noise"] = True
    reward_cfg = loaded_config.get("reward_cfg")
    _command_cfg = loaded_config.get("command_cfg")
    env_cfg["base_init_pos"] = [0.0, 0.0, 0.30]
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": args.command_x,
        "lin_vel_y_range": args.command_y,
        "ang_vel_range": args.command_ang,
    }
    reward_cfg["reward_scales"] = {}

    env = A1CPGEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        capture=args.capture,
        device=device,
        eval=True
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=device)
    resume_path = os.path.join(log_dir, "Networks",f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=device)
    max_dof_vel = env.dof_vel
    min_dof_vel = env.dof_vel

    csv_file_path = f"{test_dir}/actions_obs.csv"
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header
        csv_writer.writerow(['step', 'action', 'observation', 'foot_force_norms'])

        obs, _ = env.reset()
        with torch.no_grad():
            # while True:
            env.cam.start_recording()
            for i in tqdm.tqdm(range(500)):
                actions = policy(obs)
                obs, _, rews, dones, infos = env.step(actions)
                max_dof_vel = torch.max(max_dof_vel, env.dof_vel)
                min_dof_vel = torch.min(min_dof_vel, env.dof_vel)
                csv_writer.writerow([i, actions.cpu().numpy().tolist(), obs.cpu().numpy().tolist(), env.foot_force_norms.cpu().numpy().tolist()])
            env.cam.stop_recording(save_to_filename=f"{test_dir}/video.mp4", fps=50)

    print(f"max_dof_vel: {max_dof_vel}")
    print(f"min_dof_vel: {min_dof_vel}")

if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""

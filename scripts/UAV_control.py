import os
import sys
import time
from pathlib import Path

import airsim
import numpy as np
import cv2

# 把项目根目录加入 sys.path，方便从任意脚本运行
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from airsim_plugin.AirVLNSimulatorClientTool import AirVLNSimulatorClientTool, State
from utils.logger import logger

# 你要收集的场景名字，必须和 env_exec_path_dict 里的 key 对应，比如 "NewYorkCity"
TARGET_SCENE = "NewYorkCity"


def save_rgb_images(client: airsim.MultirotorClient, step_idx: int, save_root: str):
    """
    从多个 camera 采集 RGB 图像并保存到磁盘。
    项目里相机名一般是 frontcamera / leftcamera / rightcamera / rearcamera / downcamera。
    """
    os.makedirs(save_root, exist_ok=True)

    rgb_cams = ["frontcamera", "leftcamera", "rightcamera", "rearcamera", "downcamera"]
    requests = [
        airsim.ImageRequest(cam, airsim.ImageType.Scene, pixels_as_float=False, compress=False)
        for cam in rgb_cams
    ]

    responses = client.simGetImages(requests)

    for cam, resp in zip(rgb_cams, responses):
        img1d = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
        if img1d.size == 0:
            logger.warning(f"[step {step_idx}] camera {cam} returned empty image, skip.")
            continue

        img_rgba = img1d.reshape(resp.height, resp.width, 4)
        img_rgb = img_rgba[:, :, :3]

        cam_dir = os.path.join(save_root, cam)
        os.makedirs(cam_dir, exist_ok=True)
        out_path = os.path.join(cam_dir, f"{step_idx:06d}.png")
        cv2.imwrite(out_path, img_rgb[:, :, ::-1])  # RGBA -> BGR for OpenCV


def main():
    # 1. 构造 machines_info：告诉 ClientTool 要在哪台机器上、开哪些场景
    machines_info = [
        {
            "MACHINE_IP": "127.0.0.1",
            "SOCKET_PORT": 30000,          # 这里就是你启动 ServerTool 时的 --port
            "open_scenes": [TARGET_SCENE], # 只开一个 NewYorkCity
            "gpus": [0],                   # 用第 0 块 GPU（和你的机器对应即可）
            "MAX_SCENE_NUM": 1,            # 最多一个场景
        }
    ]

    sim_tool = AirVLNSimulatorClientTool(machines_info)

    # 2. 通过 socket 让 ServerTool 打开 AirSim 场景，并建立 MultirotorClient
    logger.info("Connecting to AirSim via AirVLNSimulatorClientTool...")
    sim_tool.run_call()  # 内部会通过 30000 端口调用 reopen_scenes，然后创建 airsim.MultirotorClient

    # 我们只开了一个场景，所以 client 在 sim_tool.airsim_clients[0][0]
    client: airsim.MultirotorClient = sim_tool.airsim_clients[0][0]
    assert client is not None, "AirSim client is None, check if environment started correctly."

    client.confirmConnection()
    logger.info("AirSim connection confirmed.")

    # 3. 控制 UAV：解锁+起飞
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()
    logger.info("Takeoff complete.")

    # 4. 初始化状态读取器 & 输出目录
    state_sensor = State(client)
    save_root = "my_new_dataset_run1"
    Path(save_root).mkdir(parents=True, exist_ok=True)

    traj = []
    num_steps = 200

    for step in range(num_steps):
        # 这里写你的控制策略：例子是沿 X 方向匀速前进
        client.moveByVelocityAsync(3, 0, 0, 0.2).join()

        # 记录当前状态到 traj
        s = state_sensor.retrieve()
        s["step"] = step
        traj.append(s)

        # 保存多相机图像
        save_rgb_images(client, step, os.path.join(save_root, "rgb"))

        time.sleep(0.05)

    # 5. 降落并释放控制
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)

    # 6. 保存轨迹为 JSON
    import json
    traj_path = os.path.join(save_root, "trajectory.json")
    with open(traj_path, "w") as f:
        json.dump(traj, f, indent=2, default=lambda o: getattr(o, "__dict__", str(o)))
    logger.info(f"Saved trajectory to {traj_path}")


if __name__ == "__main__":
    main()
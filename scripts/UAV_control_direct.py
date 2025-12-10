import os
import time
import json
from pathlib import Path

import airsim
import numpy as np
import cv2

VEHICLE_NAME = "Drone_1"
# TODO: 把这里改成你 settings.json 里真实的 ApiServerPort
API_PORT = 41451  # 例如 41451，如果不一样，请改成实际的端口

RGB_CAMERAS = ["Frontcamera", "Leftcamera", "Rightcamera", "Rearcamera", "Downcamera"]


def save_rgb_images(client: airsim.MultirotorClient, step_idx: int, save_root: str):
    """从多个相机采集 RGB 图像并保存。"""
    os.makedirs(save_root, exist_ok=True)

    requests = [
        airsim.ImageRequest(cam, airsim.ImageType.Scene, pixels_as_float=False, compress=False)
        for cam in RGB_CAMERAS
    ]
    responses = client.simGetImages(requests, vehicle_name=VEHICLE_NAME)

    for cam, resp in zip(RGB_CAMERAS, responses):
        img1d = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
        if img1d.size == 0:
            print(f"[step {step_idx}] camera {cam} returned empty image, skip.")
            continue

        img_rgba = img1d.reshape(resp.height, resp.width, 4)
        img_rgb = img_rgba[:, :, :3]

        cam_dir = os.path.join(save_root, cam)
        os.makedirs(cam_dir, exist_ok=True)
        out_path = os.path.join(cam_dir, f"{step_idx:06d}.png")
        cv2.imwrite(out_path, img_rgb[:, :, ::-1])  # RGBA -> BGR for OpenCV


def main():
    # 1. 连接到 AirSim 实例
    client = airsim.MultirotorClient(ip="127.0.0.1", port=API_PORT)
    client.confirmConnection()
    print(f"Connected to AirSim at 127.0.0.1:{API_PORT}")

    # 2. 解锁并起飞
    client.enableApiControl(True, vehicle_name=VEHICLE_NAME)
    client.armDisarm(True, vehicle_name=VEHICLE_NAME)
    client.takeoffAsync(vehicle_name=VEHICLE_NAME).join()
    print("Takeoff complete.")

    # 3. 准备保存目录
    save_root = "my_direct_dataset_run1"
    Path(save_root).mkdir(parents=True, exist_ok=True)

    traj = []
    num_steps = 200

    for step in range(num_steps):
        # 控制：沿 X 方向匀速飞行
        client.moveByVelocityAsync(3, 0, 0, 0.2, vehicle_name=VEHICLE_NAME).join()

        # 记录位姿信息
        state = client.getMultirotorState(vehicle_name=VEHICLE_NAME)
        pos = state.kinematics_estimated.position
        ori = state.kinematics_estimated.orientation
        s = {
            "step": step,
            "timestamp": state.timestamp,
            "position": [pos.x_val, pos.y_val, pos.z_val],
            "orientation": [ori.w_val, ori.x_val, ori.y_val, ori.z_val],
        }
        traj.append(s)

        # 保存相机图像
        save_rgb_images(client, step, os.path.join(save_root, "rgb"))

        time.sleep(0.05)

    # 4. 降落并释放控制
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)

    # 5. 保存轨迹 JSON
    traj_path = os.path.join(save_root, "trajectory.json")
    with open(traj_path, "w") as f:
        json.dump(traj, f, indent=2)
    print(f"Saved trajectory to {traj_path}")


if __name__ == "__main__":
    main()
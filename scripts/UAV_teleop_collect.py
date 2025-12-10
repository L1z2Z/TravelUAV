import os
import json
import time
from pathlib import Path

import airsim
import cv2
import numpy as np
import pygame  # 新增：用于键盘连续输入


# === 根据 settings/30001/settings.json 设定 ===
API_PORT = 30001          # 对应 "ApiServerPort": 30001
VEHICLE_NAME = "Drone_1"  # settings 里配置的无人机名字
CAM_NAMES = ["FrontCamera", "LeftCamera", "RightCamera", "RearCamera", "DownCamera"]

# 数据保存根目录
SAVE_ROOT = "/home/ubuntu/xrf/data/TravelUAV_mini/memory_data/NewYorkCity"
PREFIX = "NYC_"
WIDTH = 4

# 创建该条轨迹的文件夹
def create_next_folder(root=SAVE_ROOT, prefix=PREFIX, width=WIDTH):
    # 确保 ROOT 目录存在
    os.makedirs(root, exist_ok=True)

    # 列出 ROOT 目录下的所有子项
    items = os.listdir(root)

    # 找出所有符合 NYC_xxxx 规则的文件夹
    indices = []
    for name in items:
        full_path = os.path.join(root, name)
        # 只考虑文件夹，并且名字以指定前缀开头
        if os.path.isdir(full_path) and name.startswith(prefix):
            suffix = name[len(prefix):]  # 取出后面的数字部分
            # 判断后面是不是全是数字，且长度刚好等于 width
            if len(suffix) == width and suffix.isdigit():
                indices.append(int(suffix))

    # 计算下一个编号
    if indices:
        next_index = max(indices) + 1
    else:
        next_index = 0  # 第一次创建，从 0 开始

    # 格式化成带前导零的字符串，例如 0 -> "0000"
    folder_name = f"{prefix}{next_index:0{width}d}"
    folder_path = os.path.join(root, folder_name)

    # 创建文件夹
    os.makedirs(folder_path, exist_ok=False)

    return folder_path

# 保存图片
def save_multi_camera_images(client: airsim.MultirotorClient, step_idx: int, save_path: str):
    """从五个方向相机采集 RGB 和 Depth 图像并保存到磁盘。"""
    requests = []
    for cam in CAM_NAMES:
        requests.append(
            airsim.ImageRequest(cam, airsim.ImageType.Scene, pixels_as_float=False, compress=False)
        )
        requests.append(
            airsim.ImageRequest(cam, airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False)
        )

    responses = client.simGetImages(requests, vehicle_name=VEHICLE_NAME)

    for i, cam in enumerate(CAM_NAMES):
        resp_rgb = responses[2 * i]
        resp_depth = responses[2 * i + 1]

        # --- RGB 部分 ---
        img1d = np.frombuffer(resp_rgb.image_data_uint8, dtype=np.uint8)
        if img1d.size == 0:
            print(f"[step {step_idx}] camera {cam} RGB returned empty image, skip.")
            continue

        h, w = resp_rgb.height, resp_rgb.width
        expected_size = h * w * 3
        if img1d.size != expected_size:
            print(
                f"[step {step_idx}] camera {cam} RGB unexpected buffer size: "
                f"size={img1d.size}, h={h}, w={w}, expected={expected_size}. Skip this RGB frame."
            )
        else:
            img_rgb = img1d.reshape(h, w, 3)
            cam_rgb_dir = os.path.join(save_path, cam.lower())
            os.makedirs(cam_rgb_dir, exist_ok=True)
            rgb_path = os.path.join(cam_rgb_dir, f"{step_idx:06d}.png")
            cv2.imwrite(rgb_path, img_rgb[:, :, ::-1])

        # --- Depth 部分 ---
        depth_data = np.array(resp_depth.image_data_float, dtype=np.float32)
        if depth_data.size == 0:
            print(f"[step {step_idx}] camera {cam} DEPTH returned empty image, skip.")
            continue

        dh, dw = resp_depth.height, resp_depth.width
        if depth_data.size != dh * dw:
            print(
                f"[step {step_idx}] camera {cam} DEPTH unexpected buffer size: "
                f"size={depth_data.size}, h={dh}, w={dw}, expected={dh * dw}. Skip this depth frame."
            )
            continue

        depth_image = depth_data.reshape(dh, dw)
        max_depth = 100.0
        depth_clipped = np.clip(depth_image, 0, max_depth)
        depth_norm = (depth_clipped / max_depth * 255).astype(np.uint8)

        cam_depth_dir = os.path.join(save_path, cam.lower() + "_depth")
        os.makedirs(cam_depth_dir, exist_ok=True)
        depth_png_path = os.path.join(cam_depth_dir, f"{step_idx:06d}.png")
        cv2.imwrite(depth_png_path, depth_norm)


def state_to_dict(state: airsim.MultirotorState, step_idx: int) -> dict:
    pos = state.kinematics_estimated.position
    ori = state.kinematics_estimated.orientation
    lin_vel = state.kinematics_estimated.linear_velocity
    ang_vel = state.kinematics_estimated.angular_velocity

    return {
        "step": step_idx,
        "timestamp": state.timestamp,
        "position": [pos.x_val, pos.y_val, pos.z_val],
        "orientation": [ori.w_val, ori.x_val, ori.y_val, ori.z_val],
        "linear_velocity": [lin_vel.x_val, lin_vel.y_val, lin_vel.z_val],
        "angular_velocity": [ang_vel.x_val, ang_vel.y_val, ang_vel.z_val],
    }


def main():
    Path(SAVE_ROOT).mkdir(parents=True, exist_ok=True)

    # 1. 连接到 NewYorkCity 仿真
    client = airsim.MultirotorClient(ip="127.0.0.1", port=API_PORT)
    client.confirmConnection()
    print(f"Connected to AirSim at 127.0.0.1:{API_PORT}")

    # 2. 绑定 UAV 控制
    client.enableApiControl(True, vehicle_name=VEHICLE_NAME)
    client.armDisarm(True, vehicle_name=VEHICLE_NAME)
    print("enableApiControl and armDisarm done.")

    pose = client.simGetVehiclePose(vehicle_name=VEHICLE_NAME)
    pose.position.z_val = -10  # 把 z 往上提
    client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=VEHICLE_NAME)

    # 先确保仿真不是暂停状态
    client.simPause(False)
    try:
        client.moveToZAsync(-10, 3, vehicle_name=VEHICLE_NAME).join()
        print("moveToZAsync(-10) done.")
    except Exception as e:
        print("moveToZAsync ERROR:", repr(e))

    # 打印当前状态看一下
    state = client.getMultirotorState(vehicle_name=VEHICLE_NAME)
    print("Initial state after moveToZAsync:")
    print("  position:", state.kinematics_estimated.position)
    print("  collision.has_collided:", state.collision.has_collided)

    trajectory = []
    step_idx = 0

    print("控制说明：W/S 前后, A/D 左右, R/F 上下, Q/E 旋转, P 开始/结束记录, Esc 退出")
    print("使用 pygame 连续读取键盘，无需每次按回车。")

    # 初始化 pygame，用于键盘连续读取
    pygame.init()
    # 创建一个窗口以接收键盘事件和显示预览
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("UAV Teleop (WASD/RF/QE, 1-5 预览相机, P 记录, ESC 退出)")
    clock = pygame.time.Clock()

    recording = False
    prev_p_pressed = False

    running = True
    base_speed = 3.0
    duration = 0.1  # 每个时间片的持续时间（秒）越小越平滑
    yaw_rate = 10.0  # 每个时间片的 yaw 变化量（度）

    # 预览相机，默认前向
    current_preview_cam = "FrontCamera"

    while running:
        # 处理 pygame 事件（关闭窗口等）
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        # Esc 退出
        if keys[pygame.K_ESCAPE]:
            running = False

        # 相机预览切换（保持选择状态，不每帧重置）
        if keys[pygame.K_1]:
            current_preview_cam = "FrontCamera"
        if keys[pygame.K_2]:
            current_preview_cam = "LeftCamera"
        if keys[pygame.K_3]:
            current_preview_cam = "RightCamera"
        if keys[pygame.K_4]:
            current_preview_cam = "RearCamera"
        if keys[pygame.K_5]:
            current_preview_cam = "DownCamera"

        # 拉一张当前预览相机的图像并显示在 pygame 窗口
        try:
            resp = client.simGetImage(
                current_preview_cam,
                airsim.ImageType.Scene,
                vehicle_name=VEHICLE_NAME,
            )
            if resp is not None:
                img1d = np.frombuffer(resp, dtype=np.uint8)
                if img1d.size > 0:
                    img = cv2.imdecode(img1d, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        surf = pygame.surfarray.make_surface(np.rot90(img_rgb))
                        screen.blit(surf, (0, 0))
        except Exception as e:
            # 预览失败不影响控制
            pass

        # 处理 P 键：上升沿触发开始/结束记录
        p_pressed = keys[pygame.K_p]
        if p_pressed and not prev_p_pressed:
            recording = not recording
            if recording:
                save_path = create_next_folder()  # 创建新的轨迹数据文件夹
                base_speed = 1.0
                print("[P] 开始记录轨迹和图像")
                # 记录初始帧
                state = client.getMultirotorState(vehicle_name=VEHICLE_NAME)
                trajectory.append(state_to_dict(state, step_idx))
                save_multi_camera_images(client, step_idx, save_path)
                step_idx += 1
            else:
                print("[P] 结束记录，将在退出或下次按 ESC 后保存最后一帧")
                base_speed = 3.0
        prev_p_pressed = p_pressed

        vx = vy = vz = 0.0

        # WASD 水平移动（机体坐标）
        if keys[pygame.K_w]:
            vx += base_speed
        if keys[pygame.K_s]:
            vx -= base_speed
        if keys[pygame.K_a]:
            vy -= base_speed
        if keys[pygame.K_d]:
            vy += base_speed

        # R/F 垂直移动（AirSim 中 z 轴向下为正）
        if keys[pygame.K_r]:
            vz -= 5 * base_speed
        if keys[pygame.K_f]:
            vz += 5 * base_speed

        # Q/E 控制 yaw 旋转
        if keys[pygame.K_q] or keys[pygame.K_e]:
            curr_yaw = airsim.to_eularian_angles(
                client.getMultirotorState(vehicle_name=VEHICLE_NAME).kinematics_estimated.orientation
            )[2]
            if keys[pygame.K_q]:
                target_yaw = curr_yaw - np.deg2rad(yaw_rate)
            else:
                target_yaw = curr_yaw + np.deg2rad(yaw_rate)
            try:
                client.rotateToYawAsync(np.rad2deg(target_yaw), vehicle_name=VEHICLE_NAME).join()
            except Exception as e:
                print("rotateToYawAsync ERROR:", repr(e))

        # 只有当有按键时才发送控制
        if any([vx, vy, vz]):
            try:
                # 使用机体坐标系速度
                client.moveByVelocityBodyFrameAsync(
                    vx,
                    vy,
                    vz,
                    duration,
                    drivetrain=airsim.DrivetrainType.ForwardOnly,
                    yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0.0),
                    vehicle_name=VEHICLE_NAME,
                ).join()
            except Exception as e:
                print("moveByVelocityAsync ERROR:", repr(e))
            else:
                if recording:
                    state = client.getMultirotorState(vehicle_name=VEHICLE_NAME)
                    trajectory.append(state_to_dict(state, step_idx))
                    if step_idx % 10 == 0:
                        save_multi_camera_images(client, step_idx, save_path)
                    step_idx += 1

        pygame.display.flip()
        clock.tick(30)

    # 如果处于记录状态，记录结束帧
    if recording:
        state = client.getMultirotorState(vehicle_name=VEHICLE_NAME)
        trajectory.append(state_to_dict(state, step_idx))
        save_multi_camera_images(client, step_idx, save_path)

    pygame.quit()

    try:
        client.landAsync(vehicle_name=VEHICLE_NAME).join()
    except Exception as e:
        print("landAsync ERROR (can ignore):", repr(e))
    client.armDisarm(False, vehicle_name=VEHICLE_NAME)
    client.enableApiControl(False, vehicle_name=VEHICLE_NAME)

    # 8. 保存轨迹 JSON
    traj_path = os.path.join(save_path, "trajectory.json")
    with open(traj_path, "w") as f:
        json.dump(trajectory, f, indent=2)
    print(f"Saved trajectory to {traj_path}")


if __name__ == "__main__":
    main()
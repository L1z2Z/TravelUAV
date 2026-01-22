import os
import json
import time
import math
import threading
from pathlib import Path
import argparse
import airsim
import cv2
import numpy as np
import pygame

'''
-- usage --
python UAV_teleop_collect_smooth.py --env NYC --root_path "/home/liz/data/TravelUAV_data/memory_data/NewYorkCity"
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="env name")
    parser.add_argument("--root_path", type=str, required=True, help="save root path")
    return parser.parse_args()


def _ramp(current: float, target: float, max_delta: float) -> float:
    """Limit how fast a command can change (simple acceleration limiter)."""
    if target > current + max_delta:
        return current + max_delta
    if target < current - max_delta:
        return current - max_delta
    return target


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


# === 根据 settings/30001/settings.json 设定 ===
API_PORT = 30001          # 对应 "ApiServerPort": 30001
VEHICLE_NAME = "Drone_1"  # settings 里配置的无人机名字
CAM_NAMES = ["FrontCamera", "LeftCamera", "RightCamera", "RearCamera", "DownCamera"]

# 数据保存根目录
# SAVE_ROOT = "/home/liz/data/TravelUAV_data/memory_data/NewYorkCity"
WIDTH = 4


def create_next_folder(root, prefix, width=WIDTH):
    """创建一个新的轨迹目录，如 NYC_0000, NYC_0001 ..."""
    os.makedirs(root, exist_ok=True)

    items = os.listdir(root)
    indices = []
    for name in items:
        full_path = os.path.join(root, name)
        if os.path.isdir(full_path) and name.startswith(prefix):
            suffix = name[len(prefix):]
            if len(suffix) == width and suffix.isdigit():
                indices.append(int(suffix))

    next_index = max(indices) + 1 if indices else 0
    folder_name = f"{prefix}{next_index:0{width}d}"
    folder_path = os.path.join(root, folder_name)
    os.makedirs(folder_path, exist_ok=False)
    return folder_path


def save_multi_camera_images(client: airsim.MultirotorClient, step_idx: int, save_path: str):
    """从五个方向相机采集 RGB 和 Depth 图像并保存到磁盘。"""
    requests = []
    for cam in CAM_NAMES:
        requests.append(airsim.ImageRequest(cam, airsim.ImageType.Scene, pixels_as_float=False, compress=False))
        requests.append(airsim.ImageRequest(cam, airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False))

    responses = client.simGetImages(requests, vehicle_name=VEHICLE_NAME)

    for i, cam in enumerate(CAM_NAMES):
        resp_rgb = responses[2 * i]
        resp_depth = responses[2 * i + 1]

        # --- RGB ---
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
            # OpenCV 写盘需要 BGR
            cv2.imwrite(rgb_path, img_rgb[:, :, ::-1])

        # --- Depth ---
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


class ControlThread(threading.Thread):
    """固定频率控制线程：

    - 主线程只负责：读键盘/显示/记录
    - 控制线程只负责：以固定 dt 发送控制指令（不受 simGetImage / 写盘 等阻塞影响）
    - 垂直方向用 Z-hold（moveByVelocityZ*）保证松键悬停
    """

    def __init__(self, client_ctrl: airsim.MultirotorClient, shared: dict, lock: threading.Lock, stop_evt: threading.Event):
        super().__init__(daemon=True)
        self.client = client_ctrl
        self.shared = shared
        self.lock = lock
        self.stop_evt = stop_evt

    def run(self):
        with self.lock:
            hz = float(self.shared["CONTROL_HZ"])
        dt = 1.0 / hz
        cmd_duration = max(0.08, 3.0 * dt)  # 略大于控制周期，避免“断档”

        # 初始化 yaw 估计（只用于 moveByVelocityZAsync 的 fallback）
        try:
            pose = self.client.simGetVehiclePose(vehicle_name=VEHICLE_NAME)
            _, _, yaw_est = airsim.to_eularian_angles(pose.orientation)
        except Exception:
            yaw_est = 0.0

        # 初始化目标高度（NED: z 越小越高）
        try:
            st = self.client.getMultirotorState(vehicle_name=VEHICLE_NAME)
            target_z = float(st.kinematics_estimated.position.z_val)
        except Exception:
            target_z = -10.0

        vx = 0.0
        vy = 0.0
        vz_cmd = 0.0
        yaw_rate = 0.0

        next_t = time.perf_counter()

        while not self.stop_evt.is_set():
            next_t += dt

            with self.lock:
                vx_norm = float(self.shared["vx_norm"])
                vy_norm = float(self.shared["vy_norm"])
                vz_norm = float(self.shared["vz_norm"])
                yaw_norm = float(self.shared["yaw_norm"])

                max_xy = float(self.shared["max_speed_xy"])
                max_z = float(self.shared["max_speed_z"])
                max_yaw = float(self.shared["max_yaw_rate_deg_s"])

                acc_xy = float(self.shared["accel_xy"])
                acc_z = float(self.shared["accel_z"])
                acc_yaw = float(self.shared["accel_yaw"])

            tgt_vx = vx_norm * max_xy
            tgt_vy = vy_norm * max_xy
            tgt_vz = vz_norm * max_z
            tgt_yaw_rate = yaw_norm * max_yaw

            vx = _ramp(vx, tgt_vx, acc_xy * dt)
            vy = _ramp(vy, tgt_vy, acc_xy * dt)
            vz_cmd = _ramp(vz_cmd, tgt_vz, acc_z * dt)
            yaw_rate = _ramp(yaw_rate, tgt_yaw_rate, acc_yaw * dt)

            # Z-hold：把“垂直速度意图”积分成目标 z
            target_z += vz_cmd * dt
            target_z = _clamp(target_z, -200.0, -0.5)

            try:
                if hasattr(self.client, "moveByVelocityZBodyFrameAsync"):
                    self.client.moveByVelocityZBodyFrameAsync(
                        vx,
                        vy,
                        target_z,
                        cmd_duration,
                        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                        yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate),
                        vehicle_name=VEHICLE_NAME,
                    )
                else:
                    # fallback：如果没有 body-frame 版本，就近似把 body 速度旋到 world 再发
                    yaw_est += math.radians(yaw_rate) * dt
                    cos_y = math.cos(yaw_est)
                    sin_y = math.sin(yaw_est)
                    vx_w = vx * cos_y - vy * sin_y
                    vy_w = vx * sin_y + vy * cos_y
                    self.client.moveByVelocityZAsync(
                        vx_w,
                        vy_w,
                        target_z,
                        cmd_duration,
                        yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate),
                        vehicle_name=VEHICLE_NAME,
                    )
            except Exception:
                # 控制线程里不要疯狂打印，避免再次卡顿
                pass

            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                # 如果落后太多，直接重置节拍
                next_t = time.perf_counter()


class PreviewThread(threading.Thread):
    """相机拉流线程：把 simGetImage/cv2.imdecode 放到后台，避免卡控制。"""

    def __init__(self, client_preview: airsim.MultirotorClient, shared: dict, lock: threading.Lock, stop_evt: threading.Event):
        super().__init__(daemon=True)
        self.client = client_preview
        self.shared = shared
        self.lock = lock
        self.stop_evt = stop_evt

    def run(self):
        with self.lock:
            hz = float(self.shared["PREVIEW_HZ"])
        dt = 1.0 / hz
        next_t = time.perf_counter()

        while not self.stop_evt.is_set():
            next_t += dt

            with self.lock:
                cam = self.shared["preview_cam"]

            try:
                resp = self.client.simGetImage(cam, airsim.ImageType.Scene, vehicle_name=VEHICLE_NAME)
            except Exception:
                resp = None

            if resp:
                try:
                    img1d = np.frombuffer(resp, dtype=np.uint8)
                    if img1d.size > 0:
                        img = cv2.imdecode(img1d, cv2.IMREAD_UNCHANGED)
                        if img is not None:
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            with self.lock:
                                self.shared["preview_rgb"] = img_rgb
                except Exception:
                    pass

            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()


def main():
    args = parse_args()
    global ENV_NAME, SAVE_ROOT
    ENV_NAME = args.env
    SAVE_ROOT = args.root_path

    Path(SAVE_ROOT).mkdir(parents=True, exist_ok=True)

    # 控制 client（只在控制线程使用）
    client_ctrl = airsim.MultirotorClient(ip="127.0.0.1", port=API_PORT)
    client_ctrl.confirmConnection()
    print(f"Connected to AirSim at 127.0.0.1:{API_PORT}")

    client_ctrl.enableApiControl(True, vehicle_name=VEHICLE_NAME)
    client_ctrl.armDisarm(True, vehicle_name=VEHICLE_NAME)
    print("enableApiControl and armDisarm done.")

    # 把飞机抬到空中
    pose = client_ctrl.simGetVehiclePose(vehicle_name=VEHICLE_NAME)
    pose.position.z_val = -10
    pose.position.y_val = 0
    pose.position.x_val = 0
    #pose.position.y_val = -10
    #pose.position.x_val = 0
    client_ctrl.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=VEHICLE_NAME)


    client_ctrl.simPause(False)
    try:
        client_ctrl.moveToZAsync(-10, 3, vehicle_name=VEHICLE_NAME).join()
    except Exception as e:
        print("moveToZAsync ERROR:", repr(e))

    # IO client（只在主线程使用：getState / simGetImages / 写盘）
    client_io = airsim.MultirotorClient(ip="127.0.0.1", port=API_PORT)
    client_io.confirmConnection()

    # 预览 client（只在预览线程使用）
    client_preview = airsim.MultirotorClient(ip="127.0.0.1", port=API_PORT)
    client_preview.confirmConnection()

    # 初始化 pygame
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("UAV Teleop (WASD/RF/QE, 1-5 预览相机, C 打印坐标, P 记录, ESC 退出)")
    clock = pygame.time.Clock()

    print("控制说明：W/S 前后, A/D 左右, R/F 上下, Q/E 旋转, C 打印坐标, P 开始/结束记录, Esc 退出")

    shared = {
        "CONTROL_HZ": 50,
        "PREVIEW_HZ": 10,

        # 输入轴（归一化 -1~1）
        "vx_norm": 0.0,
        "vy_norm": 0.0,
        "vz_norm": 0.0,   # NED: +down
        "yaw_norm": 0.0,

        # 上限与平滑参数
        "max_speed_xy": 1.5,
        "max_speed_z": 2.0,
        "max_yaw_rate_deg_s": 30.0,
        "accel_xy": 2.0,
        "accel_z": 2.0,
        "accel_yaw": 90.0,

        # 预览
        "preview_cam": "FrontCamera",
        "preview_rgb": None,
    }

    lock = threading.Lock()
    stop_evt = threading.Event()

    ctrl_thread = ControlThread(client_ctrl, shared, lock, stop_evt)
    preview_thread = PreviewThread(client_preview, shared, lock, stop_evt)
    ctrl_thread.start()
    preview_thread.start()

    # 记录参数
    RECORD_HZ = 10
    IMAGE_HZ = 2
    record_interval = 1.0 / RECORD_HZ
    image_stride = max(1, int(RECORD_HZ / IMAGE_HZ))
    last_record_t = 0.0

    recording = False
    prev_p_pressed = False
    save_path = None
    trajectory = []
    step_idx = 0

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False

        # --- 预览相机选择 ---
        cam = None
        if keys[pygame.K_1]:
            cam = "FrontCamera"
        elif keys[pygame.K_2]:
            cam = "LeftCamera"
        elif keys[pygame.K_3]:
            cam = "RightCamera"
        elif keys[pygame.K_4]:
            cam = "RearCamera"
        elif keys[pygame.K_5]:
            cam = "DownCamera"

        # --- C 打印坐标 ---
        if keys[pygame.K_c]:
            st = client_io.getMultirotorState(vehicle_name=VEHICLE_NAME)
            print(st.kinematics_estimated.position)
            time.sleep(0.2)  # 防止多次打印

        # --- P 开始/结束记录（上升沿触发） ---
        p_pressed = keys[pygame.K_p]
        if p_pressed and not prev_p_pressed:
            recording = not recording
            if recording:
                save_path = create_next_folder(root=SAVE_ROOT, prefix=ENV_NAME + "_", width=WIDTH)
                print("[P] 开始记录轨迹和图像")
                # 录制时降低速度上限，更好采集
                with lock:
                    shared["max_speed_xy"] = 1.0
                    shared["max_speed_z"] = 1.0

                # 立刻记录一帧
                trajectory = []
                step_idx = 0
                st = client_io.getMultirotorState(vehicle_name=VEHICLE_NAME)
                trajectory.append(state_to_dict(st, step_idx))
                save_multi_camera_images(client_io, step_idx, save_path)
                step_idx += 1
                last_record_t = time.time()
            else:
                print("[P] 结束记录")
                with lock:
                    shared["max_speed_xy"] = 1.5
                    shared["max_speed_z"] = 2.0
        prev_p_pressed = p_pressed

        # --- 键盘 -> 归一化输入（-1, 0, +1） ---
        vx_norm = (1.0 if keys[pygame.K_w] else 0.0) - (1.0 if keys[pygame.K_s] else 0.0)
        vy_norm = (1.0 if keys[pygame.K_d] else 0.0) - (1.0 if keys[pygame.K_a] else 0.0)
        # NED: +down，所以按 R(上升) 是负，按 F(下降) 是正
        vz_norm = (1.0 if keys[pygame.K_f] else 0.0) - (1.0 if keys[pygame.K_r] else 0.0)
        yaw_norm = (1.0 if keys[pygame.K_e] else 0.0) - (1.0 if keys[pygame.K_q] else 0.0)

        # 对角线归一（避免 W+D 比单方向更快）
        xy_len = math.hypot(vx_norm, vy_norm)
        if xy_len > 1.0:
            vx_norm /= xy_len
            vy_norm /= xy_len

        with lock:
            shared["vx_norm"] = vx_norm
            shared["vy_norm"] = vy_norm
            shared["vz_norm"] = vz_norm
            shared["yaw_norm"] = yaw_norm
            if cam is not None:
                shared["preview_cam"] = cam
            rgb = shared["preview_rgb"]

        # --- 绘制 ---
        screen.fill((0, 0, 0))
        if rgb is not None:
            # 关键修复：不要 np.rot90（rot90 会隐式 flip 导致镜像），用 swapaxes 即可
            arr = np.ascontiguousarray(rgb.swapaxes(0, 1))
            surf = pygame.surfarray.make_surface(arr)
            surf = pygame.transform.smoothscale(surf, screen.get_size())
            screen.blit(surf, (0, 0))

        pygame.display.flip()
        clock.tick(60)

        # --- 记录（降频） ---
        now_t = time.time()
        if recording and save_path is not None:
            if (now_t - last_record_t) >= record_interval:
                last_record_t = now_t
                st = client_io.getMultirotorState(vehicle_name=VEHICLE_NAME)
                trajectory.append(state_to_dict(st, step_idx))
                if (step_idx % image_stride) == 0:
                    save_multi_camera_images(client_io, step_idx, save_path)
                step_idx += 1

    # 退出：停线程
    stop_evt.set()
    ctrl_thread.join(timeout=1.0)
    preview_thread.join(timeout=1.0)

    pygame.quit()

    # 如果处于记录状态，补最后一帧
    if recording and save_path is not None:
        st = client_io.getMultirotorState(vehicle_name=VEHICLE_NAME)
        trajectory.append(state_to_dict(st, step_idx))
        save_multi_camera_images(client_io, step_idx, save_path)

    try:
        client_ctrl.landAsync(vehicle_name=VEHICLE_NAME).join()
    except Exception:
        pass
    client_ctrl.armDisarm(False, vehicle_name=VEHICLE_NAME)
    client_ctrl.enableApiControl(False, vehicle_name=VEHICLE_NAME)

    if save_path is not None:
        traj_path = os.path.join(save_path, "trajectory.json")
        with open(traj_path, "w") as f:
            json.dump(trajectory, f, indent=2)
        print(f"Saved trajectory to {traj_path}")
    else:
        print("No trajectory saved (recording never started).")


if __name__ == "__main__":
    main()

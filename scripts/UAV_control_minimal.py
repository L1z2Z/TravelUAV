# scripts/UAV_control_minimal.py
import airsim
import time
import json
import numpy as np
import cv2
import os
from pathlib import Path

API_PORT = 41451

def main():
    client = airsim.MultirotorClient(ip="127.0.0.1", port=API_PORT)
    client.confirmConnection()
    print("Connected")

    # 不传 vehicle_name，使用默认载具
    client.enableApiControl(True)
    client.armDisarm(True)

    try:
        client.takeoffAsync().join()
        print("Takeoff done")
    except Exception as e:
        print("takeoffAsync ERROR:", repr(e))

    try:
        client.moveToZAsync(10, 2).join()  # 2 m/s 上升速度
        print("moveToZAsync(10) done")
    except Exception as e:
        print("moveToZAsync ERROR:", repr(e))

    # 试着只飞几步，不采图，先看仿真里有没有 UAV 动起来
    for i in range(10):
        try:
            client.moveByVelocityAsync(2, 0, 0, 1.0).join()
            print("step", i, "ok")
        except Exception as e:
            print("moveByVelocityAsync ERROR:", repr(e))
        time.sleep(0.1)

    # 保存一张图（默认相机）
    try:
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, pixels_as_float=False, compress=False)
        ])
        resp = responses[0]
        img1d = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
        if img1d.size > 0:
            img_rgba = img1d.reshape(resp.height, 192, 4)
            img_rgb = img_rgba[:, :, :3]
            Path("minimal_out").mkdir(parents=True, exist_ok=True)
            cv2.imwrite("minimal_out/view.png", img_rgb[:, :, ::-1])
            print("Saved minimal_out/view.png")
        else:
            print("simGetImages returned empty buffer")
    except Exception as e:
        print("simGetImages ERROR:", repr(e))

    # 尝试读取当前状态
    try:
        state = client.getMultirotorState()
        print("Multirotor state:", state)
    except Exception as e:
        print("getMultirotorState ERROR:", repr(e))

    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)

if __name__ == "__main__":
    main()
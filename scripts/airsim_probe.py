import airsim

API_PORT = 41451  # 你已经确认的端口

def main():
    client = airsim.VehicleClient(ip="127.0.0.1", port=API_PORT)
    client.confirmConnection()
    print("Connected to AirSim")

    # 1. 尝试在不指定 vehicle_name 的情况下开启 API 控制
    try:
        client.enableApiControl(True)
        print("enableApiControl(True) without vehicle_name: OK")
    except Exception as e:
        print("enableApiControl(True) without vehicle_name: ERROR:", repr(e))

    # 2. 尝试 getVehiclePose
    try:
        pose = client.simGetVehiclePose()
        print("simGetVehiclePose():", pose)
    except Exception as e:
        print("simGetVehiclePose(): ERROR:", repr(e))

    # 3. 列出场景里带有 "Drone" / "Vehicle" 的对象名称（选做）
    try:
        objs = client.simListSceneObjects()
        candidates = [o for o in objs if "Drone" in o or "drone" in o or "Vehicle" in o or "UAV" in o]
        print("Possible vehicle-like scene objects:", candidates[:50])
    except Exception as e:
        print("simListSceneObjects(): ERROR:", repr(e))

    # 4. 尝试关闭 API 控制
    try:
        client.enableApiControl(False)
        print("enableApiControl(False) without vehicle_name: OK")
    except Exception as e:
        print("enableApiControl(False) without vehicle_name: ERROR:", repr(e))


if __name__ == "__main__":
    main()
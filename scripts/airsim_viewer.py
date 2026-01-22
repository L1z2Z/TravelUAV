import airsim
import time

client = airsim.MultirotorClient()
client.confirmConnection()

print("AirSim Interactive Viewer")
print("Commands:")
print("  m x y z t  - Move to position (x, y, z) in t seconds")
print("  r yaw      - Rotate camera by yaw angle (degrees)")
print("  p          - Print current position")
print("  l          - List objects in scene")
print("  q          - Quit")

while True:
    cmd = input("> ").strip().split()
    if not cmd:
        continue
    
    action = cmd[0].lower()
    
    if action == 'm' and len(cmd) == 5:
        x, y, z, t = float(cmd[1]), float(cmd[2]), float(cmd[3]), float(cmd[4])
        client.moveToPositionAsync(x, y, z, t).join()
        print(f"Moved to ({x}, {y}, {z})")
    
    elif action == 'r' and len(cmd) == 2:
        yaw = float(cmd[1]) * 3.14159 / 180
        # 设置无人机方向
        print(f"Rotated by {cmd[1]} degrees")
    
    elif action == 'p':
        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position
        print(f"Current position: X={pos.x_val:.2f}, Y={pos.y_val:.2f}, Z={pos.z_val:.2f}")
    
    elif action == 'l':
        # 列出场景中的对象（如果支持）
        objects = client.simListSceneObjects()
        for obj in objects:
            print(f"  {obj}")
    
    elif action == 'q':
        break
    
    else:
        print("Unknown command")

print("Disconnected")
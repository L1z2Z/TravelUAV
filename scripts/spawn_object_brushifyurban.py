#!/usr/bin/env python3
"""Spawn one object into a running AirSim scene via AirVLNSimulatorClientTool.setObjects.

Assumptions:
- You already launched the Unreal/AirSim scene (e.g., BrushifyUrban) with settings that expose
  the AirSim API on port 30001.
- AirSim is reachable at 127.0.0.1:30001.

This script does NOT talk to AirVLNSimulatorServerTool; it directly connects to the running
AirSim API port and invokes setObjects().
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import airsim

# airsim_plugin/ is a plain folder (not a Python package). Add it to sys.path so
# we can import AirVLNSimulatorClientTool.py directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "airsim_plugin"))

from AirVLNSimulatorClientTool import AirVLNSimulatorClientTool


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument("--scene", default="Japanese_Street")
    args = parser.parse_args()

    # Minimal machines_info just to satisfy AirVLNSimulatorClientTool's constructor.
    machines_info = [
        {
            "MACHINE_IP": args.ip,
            "SOCKET_PORT": 30000,
            "open_scenes": [args.scene],
            "gpus": [0],
            "MAX_SCENE_NUM": 1,
        }
    ]
    tool = AirVLNSimulatorClientTool(machines_info)

    client = airsim.MultirotorClient(ip=args.ip, port=args.port, timeout_value=300)
    client.confirmConnection()

    # Bind this already-running AirSim instance into the tool.
    tool.airsim_clients = [[client]]

    asset_name = "AE_Signboards_03"
    asset_name1 = "AE_Signboards_05"
    asset_name2 = "AE_Bicycle_01"

    # Pose: [0, 0, 0] in AirSim NED world coordinates.
    pose = airsim.Pose(
        airsim.Vector3r(0.0, 0.0, 0.0),
        airsim.Quaternionr(0.0, 0.0, 0.0, 1.0),
    )
    pose1 = airsim.Pose(
        airsim.Vector3r(1.0, 0.0, 0.0),
        airsim.Quaternionr(0.0, 0.0, 0.0, 1.0),
    )
    pose2 = airsim.Pose(
        airsim.Vector3r(2.0, 0.0, 0.0),
        airsim.Quaternionr(0.0, 0.0, 0.0, 1.0),
    )
    pose3 = airsim.Pose(
        airsim.Vector3r(3.0, 0.0, 0.0),
        airsim.Quaternionr(0.0, 0.0, 0.0, 1.0),
    )

    # Scale: chosen as 1x (no scaling). Adjust if the spawned sign is too big/small.
    scale = airsim.Vector3r(1.0, 1.0, 1.0)

    object_list = [
        {
            "asset_name": asset_name,
            "pose": pose,
            "scale": scale,
        },
        {
            "asset_name": asset_name1,
            "pose": pose1,
            "scale": scale,
        },
        {
            "asset_name": asset_name2,
            "pose": pose2,
            "scale": scale,
        }
    ]

    ok = tool.setObjects(object_list)
    print("spawn success:", ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

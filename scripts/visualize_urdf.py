#!/usr/bin/env python3
"""Visualize R1Pro URDF with T265 frames using PyBullet GUI."""

import pybullet as pb
import pybullet_data
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="Visualize R1Pro URDF")
    parser.add_argument("--urdf", default="/scr/luey/brs-ctrl/assets/robot/r1_pro/r1_pro.urdf")
    args = parser.parse_args()

    # Connect to GUI
    client = pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Load ground plane
    pb.loadURDF("plane.urdf")

    # Load robot URDF
    robot_id = pb.loadURDF(
        args.urdf,
        [0, 0, 0.5],
        useFixedBase=True,
        physicsClientId=client,
    )

    # Set camera view
    pb.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.5]
    )

    # Print T265 frame info
    num_joints = pb.getNumJoints(robot_id)
    print(f"\nLoaded URDF with {num_joints} joints")
    print("\nT265 related links:")
    for i in range(num_joints):
        info = pb.getJointInfo(robot_id, i)
        link_name = info[12].decode('utf-8')
        if 't265' in link_name.lower():
            ls = pb.getLinkState(robot_id, i)
            pos = ls[0]
            print(f"  {link_name}: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    print("\nVisualization running. Press Ctrl+C or close window to exit.")
    print("Use mouse to rotate view, scroll to zoom.")

    try:
        while pb.isConnected():
            pb.stepSimulation()
            time.sleep(1/240)
    except KeyboardInterrupt:
        pass
    finally:
        pb.disconnect()
        print("Closed.")

if __name__ == "__main__":
    main()

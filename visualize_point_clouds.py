#!/usr/bin/env python3
"""Visualize radar point clouds with poses using Open3D."""

import open3d as o3d
import numpy as np
from pathlib import Path
import sys


def main():
    # Parse args the simple way
    pc_dir = "data/assembled_radar_pointclouds"
    pose_file = "data/poses_radar.txt"
    max_clouds = None
    voxel_size = 0.1
    show_traj = True

    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--max" and i + 1 < len(sys.argv) - 1:
            max_clouds = int(sys.argv[i + 2])
        elif arg == "--voxel" and i + 1 < len(sys.argv) - 1:
            voxel_size = float(sys.argv[i + 2])
        elif arg == "--no-traj":
            show_traj = False

    # Load poses: each line is 12 floats (3x4 matrix)
    poses = []
    for line in open(pose_file):
        vals = [float(x) for x in line.split()]
        if len(vals) == 12:
            pose = np.eye(4)
            pose[:3, :] = np.array(vals).reshape(3, 4)
            poses.append(pose)

    # Load point clouds
    pc_files = sorted(Path(pc_dir).glob("*.pcd"))
    if max_clouds:
        pc_files = pc_files[:max_clouds]

    n = min(len(pc_files), len(poses))
    pc_files, poses = pc_files[:n], poses[:n]

    print(f"Loading {n} point clouds from {pc_dir}")

    # Process each point cloud
    combined = o3d.geometry.PointCloud()
    centers = []

    for i, (pcd_file, pose) in enumerate(zip(pc_files, poses)):
        pcd = o3d.io.read_point_cloud(str(pcd_file))
        if len(pcd.points) == 0:
            continue

        pcd.transform(pose)

        if voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size)

        # Color: blue -> red gradient
        t = i / n
        pcd.paint_uniform_color([t, 0.3, 1.0 - t])

        combined += pcd
        centers.append(np.asarray(pcd.points).mean(axis=0))

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n}")

    print(f"Total points: {len(combined.points)}")

    # Build geometry list
    geometries = [combined]

    # Add trajectory
    if show_traj and len(centers) > 1:
        traj = o3d.geometry.LineSet()
        traj.points = o3d.utility.Vector3dVector(centers)
        traj.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(centers) - 1)])
        traj.paint_uniform_color([1, 0, 0])
        geometries.append(traj)

    # Add coordinate frame
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))

    # Show it
    o3d.visualization.draw_geometries(geometries, window_name="3QFP", width=1920, height=1080)


if __name__ == "__main__":
    main()

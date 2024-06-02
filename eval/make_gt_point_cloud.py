import open3d as o3d
import numpy as np
from scipy.linalg import inv
import csv
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
import os
from natsort import natsorted
from tqdm import tqdm

def read_poses_file(filename, calibration):
    """
        read pose file (with the kitti format)
    """
    pose_file = open(filename)

    poses = []

    Tr = calibration["Tr"]
    Tr_inv = inv(Tr)

    for line in pose_file:
        values = [float(v) for v in line.strip().split()]
        # For the Apollo dataset
        if len(values) < 12:
            pose = np.zeros((4, 4))
            pose[0:3, 3] = values[2:5]
            q = np.array(values[5:9])
            r = R.from_quat(q).as_matrix()
            pose[0:3, 0:3] = r
            pose[3, 3] = 1.0
        else:
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

        poses.append(
            np.matmul(Tr_inv, np.matmul(pose, Tr))
        )  # lidar pose in world frame

    pose_file.close()
    return poses

def read_calib_file(filename):
    """
        read calibration file (with the kitti format)
        returns -> dict calibration matrices as 4*4 numpy arrays
    """
    calib = {}
    calib_file = open(filename)
    key_num = 0

    for line in calib_file:
        # print(line)
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]
        pose = np.zeros((4, 4))

        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()
    return calib

gt_dir = "/media/shuo/T7/mai_city/mai_city/ply/sequences/02/velodyne/"
calib_file = "/media/shuo/T7/mai_city/mai_city/ply/sequences/02/calib.txt"
pose_file = "/media/shuo/T7/mai_city/mai_city/ply/sequences/02/poses.txt"
start_frame = 0
end_frame = 100
every_frame = 1
first_frame_as_origin = True

# processing
min_range = 1.5
min_z = -10.0
max_z = 30.0
pc_radius = 50.0  # distance filter for each frame
rand_downsample = False # use random or voxel downsampling
vox_down_m = 0.05

calibration = read_calib_file(calib_file)
poses = read_poses_file(pose_file, calibration)

begin_pose_inv = np.eye(4)
if first_frame_as_origin:
    begin_pose_inv = inv(poses[start_frame])

total_pc_count = natsorted(os.listdir(gt_dir))
bbx_min = np.array([-pc_radius, -pc_radius, min_z])
bbx_max = np.array([pc_radius, pc_radius, max_z])
bbx = o3d.geometry.AxisAlignedBoundingBox(bbx_min, bbx_max)

global_pcd = o3d.geometry.PointCloud()
for frame_id in tqdm(range(len(total_pc_count))):
        if (frame_id < start_frame or frame_id > end_frame or \
            frame_id % every_frame != 0):
            continue
        print("Processing frame: ", frame_id)
        # read point cloud
        pc_file = gt_dir + str(frame_id).zfill(5) + ".ply"
        pcd = o3d.io.read_point_cloud(pc_file)

        # box crop
        pcd = pcd.crop(bbx)

        # filter the distance less than min_range_m
        points = np.asarray(pcd.points)
        points = points[np.linalg.norm(points, axis=1) >= min_range]

        pc_out = o3d.geometry.PointCloud()
        pc_out.points = o3d.utility.Vector3dVector(points)

        # transform
        pose = np.matmul(begin_pose_inv, poses[frame_id])
        pc_out.transform(pose)
        global_pcd += pc_out
# save the point cloud, the file name includes start, end, every frame
save_gt_dir = "/media/shuo/T7/mai_city/mai_city/ply/sequences/02/gt/"
gt_file = save_gt_dir + str(start_frame) + "_" + str(end_frame) + "_" + str(every_frame) + ".ply"
o3d.io.write_point_cloud(gt_file, global_pcd)
print("Saved to: ", gt_file)





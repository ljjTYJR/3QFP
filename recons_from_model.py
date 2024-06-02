# reconstruct the environment from the saved model.
import sys
import numpy as np
from numpy.linalg import inv, norm
from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from shutil import copyfile

from utils.config import Config
from utils.tools import *
from utils.loss import *
from utils.mesher import Mesher
from utils.visualizer import MapVisualizer, random_color_table
from model.feature_octree import FeatureOctree
from model.feature_plane import FeaturePlane
from model.fea_encoder import FeatureEncoder
from model.decoder import Decoder
from dataset.lidar_dataset import LiDARDataset


parent_dir = "/home/shuo/projects/shuo/implicit_reconstruction/SHINE_mapping/experiments/maicity_batch_2023-09-11_12-20-57_fea_plane_0_100_4/"
model_path = parent_dir + "model/model_iter_20000.pth"

# load the model
config = Config()
config_file = parent_dir + "config.yaml"
config.load(config_file)
# run_path = setup_experiment(config)
dev = config.device
fea_enc = FeatureEncoder(config).fea_encoder
geo_mlp = Decoder(config, is_geo_encoder=True, is_time_conditioned=config.time_conditioned, in_dim=fea_enc.out_dim)
loaded_model = torch.load(model_path)
# print model dict
for key in loaded_model.keys():
    print(key)
geo_mlp.load_state_dict(loaded_model["geo_decoder"])
fea_enc = loaded_model["feature_enc"]
iters = int(loaded_model["iters"])
geo_mlp.eval()
fea_enc.eval()
print(iters)

sem_mlp = None
# Mesher
mesher = Mesher(config, fea_enc, geo_mlp, sem_mlp)

mesh_path = parent_dir + 'mesh/mesh_iter_' + str(iters+1) + ".ply"
map_path = parent_dir + 'map/sdf_map_iter_' + str(iters+1) + '_ts_' + str(0) + ".ply"
cur_mesh = mesher.recon_bbx_mesh(dataset.map_bbx, config.mc_res_m, mesh_path, map_path, config.save_map, config.semantic_on)
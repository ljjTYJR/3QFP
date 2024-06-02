import torch
import torch.nn as nn

import open3d as o3d
import copy

import time
from tqdm import tqdm
import kaolin as kal
import numpy as np

from collections import defaultdict

import multiprocessing

from utils.config import Config

from .encodings import FrequencyEncoder, tiny_cuda_nn_get_encoder, FFEncoder

class FeaturePlane(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        # first, still build the octree to generate the mask then
        # [0 1 2 3 ... max_level-1 max_level], 0 level is the root, which have 8 corners.
        self.max_level = config.tree_level_world
        self.leaf_vox_size = config.leaf_vox_size
        self.featured_level_num = config.tree_level_feat
        self.free_level_num = self.max_level - self.featured_level_num + 1
        self.feature_dim = config.feature_dim
        self.feature_std = config.feature_std
        self.polynomial_interpolation = config.poly_int_on
        self.device = config.device

        self.use_fixed_encoding = config.use_fixed_encoding
        self.use_features = config.use_features
        self.n_bins = config.n_bins
        self.n_freq = config.n_freq
        self.gaussian_scale = config.gaussian_scale
        self.embedding_size = config.embedding_size
        self.fixed_enc_type = config.fixed_enc_type
        self.out_dim = 0
        self.fea_dim = 0
        self.input_ch_pos = 0
        self.fixed_enc_type = config.fixed_enc_type
        if self.use_fixed_encoding:
            # self.pos_encoder, self.input_ch_pos = tiny_cuda_nn_get_encoder(self.fixed_enc_type, n_bins=self.n_bins, n_frequencies=self.n_freq)
            if self.fixed_enc_type == 'freq_nn':
                self.pos_encoder, self.input_ch_pos = tiny_cuda_nn_get_encoder(self.fixed_enc_type, n_bins=self.n_bins, n_frequencies=self.n_freq)
            else:
                if self.fixed_enc_type == 'freq':
                    self.pos_encoder = FrequencyEncoder(self.n_freq, self.n_bins, 3, 2)
                elif self.fixed_enc_type == 'ff':
                    self.pos_encoder = FFEncoder(self.gaussian_scale, self.embedding_size, 3)
                self.input_ch_pos = self.pos_encoder.fea_dim()
        if self.use_features:
            self.fea_dim = self.feature_dim * self.featured_level_num
        self.out_dim = self.input_ch_pos + self.fea_dim

        self.corners_lookup_tables = [] # from corner morton to corner index (top-down)
        self.nodes_lookup_tables = []
        self.xy_nodes_lookup_tables = []
        self.xy_corners_lookup_tables = []
        self.xz_nodes_lookup_tables = []
        self.xz_corners_lookup_tables = []
        self.yz_nodes_lookup_tables = []
        self.yz_corners_lookup_tables = []

        for l in range(self.max_level+1):
            self.corners_lookup_tables.append({})
            self.nodes_lookup_tables.append({}) # actually the same speed as below
            self.xy_nodes_lookup_tables.append({})
            self.xy_corners_lookup_tables.append({})
            self.xz_nodes_lookup_tables.append({})
            self.xz_corners_lookup_tables.append({})
            self.yz_nodes_lookup_tables.append({})
            self.yz_corners_lookup_tables.append({})

        if self.featured_level_num < 1:
            raise ValueError('No level with grid features!')

        self.xy_hier_features = nn.ParameterList([])
        self.xz_hier_features = nn.ParameterList([])
        self.yz_hier_features = nn.ParameterList([])

        self.xy_hier_features_last = []
        self.xz_hier_features_last = []
        self.yz_hier_features_last = []

        self.to(config.device)

    def forward(self, x):
        pass

    def set_zero(self):
        with torch.no_grad():
            for n in range(len(self.xy_hier_features)):
                self.xy_hier_features[n][-1] = torch.zeros(1,self.feature_dim)
            for n in range(len(self.xz_hier_features)):
                self.xz_hier_features[n][-1] = torch.zeros(1,self.feature_dim)
            for n in range(len(self.yz_hier_features)):
                self.yz_hier_features[n][-1] = torch.zeros(1,self.feature_dim)

    def get_indices(self, x):
        self.hier_plane_indices = [] # level: from finest to coarsest
        for i in range(self.featured_level_num):        #[0,1,2]
            cur_lvl = self.max_level - i                # [max_level, max_level-1, max_level-2]
            """
            NOTE One minor thing is that, for unsampled points, the calculated indices are -1 of course;
            However, for some unsampled points `p`, even though they "should" be unsampled, but the corresponding projected planes
            are sampled, so the extracted indices are not -1.
            When querying indices, these points will have non-zero features, which is not reasonable.
            """
            points = kal.ops.spc.quantize_points(x, cur_lvl)
            points_morton = kal.ops.spc.points_to_morton(points).cpu().numpy().tolist()
            default_ = False
            in_the_table = torch.tensor([self.nodes_lookup_tables[cur_lvl].get(m, default_) for m in points_morton]).bool().to(self.device)
            not_in_the_table = torch.logical_not(in_the_table)

            # check whether points_morton in the table
            xy_points = copy.deepcopy(points); xy_points[:, 2] = 0; xy_points_morton = kal.ops.spc.points_to_morton(xy_points).cpu().numpy().tolist()
            xz_points = copy.deepcopy(points); xz_points[:, 1] = 0; xz_points_morton = kal.ops.spc.points_to_morton(xz_points).cpu().numpy().tolist()
            yz_points = copy.deepcopy(points); yz_points[:, 0] = 0; yz_points_morton = kal.ops.spc.points_to_morton(yz_points).cpu().numpy().tolist()
            fea_nunamed = [-1 for j in range(4)] # for unsampled points, the indices are `-1`
            # get the corner indices by the plane point morton
            xy_indices = [self.xy_nodes_lookup_tables[cur_lvl].get(m, fea_nunamed) for m in xy_points_morton]
            xz_indices = [self.xz_nodes_lookup_tables[cur_lvl].get(m, fea_nunamed) for m in xz_points_morton]
            yz_indices = [self.yz_nodes_lookup_tables[cur_lvl].get(m, fea_nunamed) for m in yz_points_morton]
            # NOTE for any point, as long as it includes `-1` indices, it means that the point does not correspond to any voxel. But we do not optimize free-space points, we only
            # sample surface points!
            indices_torch = torch.tensor([xy_indices, xz_indices, yz_indices]).to(self.device) # the shape should be (3, N, 4), which corresponds to `xy`, `xz`, `yz` planes
            indices_torch[:, not_in_the_table, :] = -1 # set the indices of unsampled points to -1
            self.hier_plane_indices.append(indices_torch)
        return self.hier_plane_indices

    def interpolat(self, x, cur_lvl, type: str='xy'):
        'Default: the linear interpolation is used'
        coords = ((2**cur_lvl)*(x*0.5+0.5)) # From [-1,1] -> [0,1] -> [0, 2^cur_lvl]; the coords correspond to the constructed octree
        d_coords = torch.frac(coords)
        if type == 'xy':
            xy_d_coords = d_coords[:, 0:2]
            tx = xy_d_coords[:, 0]; ty = xy_d_coords[:, 1]
            _1_tx = 1 - tx; _1_ty = 1 - ty
            c0 = _1_tx * _1_ty; c1 = _1_tx * ty; c2 = tx * _1_ty; c3 = tx * ty
        elif type == 'xz':
            xz_d_coords = d_coords[:, [0, 2]]
            tx = xz_d_coords[:, 0]; tz = xz_d_coords[:, 1]
            _1_tx = 1 - tx; _1_tz = 1 - tz
            c0 = _1_tx * _1_tz; c1 = _1_tx * tz; c2 = tx * _1_tz; c3 = tx * tz
        elif type == 'yz':
            yz_d_coords = d_coords[:, 1:3]
            ty = yz_d_coords[:, 0]; tz = yz_d_coords[:, 1]
            _1_ty = 1 - ty; _1_tz = 1 - tz
            c0 = _1_ty * _1_tz; c1 = _1_ty * tz; c2 = ty * _1_tz; c3 = ty * tz
        else:
            raise ValueError('Unknown plane type!')
        return torch.stack([c0, c1, c2, c3], dim=1)

    def query_fea_by_indices(self, x, indices): # indices:finest->coarsest
        sum_features = torch.zeros(x.shape[0], self.out_dim).to(self.device)
        if self.use_fixed_encoding and self.use_features:
            for i in range(self.featured_level_num): # from finest to coarsest
                cur_lvl = self.max_level - i
                fea_lvl = self.featured_level_num - i - 1
                xy_indices = indices[i][0]; xy_fea_cor = self.xy_hier_features[fea_lvl][xy_indices]
                xz_indices = indices[i][1]; xz_fea_cor = self.xz_hier_features[fea_lvl][xz_indices]
                yz_indices = indices[i][2]; yz_fea_cor = self.yz_hier_features[fea_lvl][yz_indices]
                xy_coeffts = self.interpolat(x, cur_lvl, 'xy')
                xz_coeffts = self.interpolat(x, cur_lvl, 'xz')
                yz_coeffts = self.interpolat(x, cur_lvl, 'yz')
                xy_fea = torch.einsum('ijk,ij->ik', xy_fea_cor, xy_coeffts)
                xz_fea = torch.einsum('ijk,ij->ik', xz_fea_cor, xz_coeffts)
                yz_fea = torch.einsum('ijk,ij->ik', yz_fea_cor, yz_coeffts)

                # Set masks for free free space points
                xy_masks = xy_indices >= 0; xy_masks_c = torch.all(xy_masks, dim=1)
                xz_masks = xz_indices >= 0; xz_masks_c = torch.all(xz_masks, dim=1)
                yz_masks = yz_indices >= 0; yz_masks_c = torch.all(yz_masks, dim=1)
                # all `true` means valid points
                masks = xy_masks_c & xz_masks_c & yz_masks_c
                _masks = torch.logical_not(masks)
                lvl_fea = xy_fea + xz_fea + yz_fea
                # the invalid points are set to zero.
                lvl_fea[_masks] = 0.
                sum_features[:, fea_lvl*self.feature_dim:(fea_lvl+1)*self.feature_dim] = lvl_fea
            pos_enc = self.pos_encoder(x)
            sum_features[:, -self.input_ch_pos:] = pos_enc
        elif self.use_fixed_encoding:
            pos_enc = self.pos_encoder(x)
            sum_features[:, -self.input_ch_pos:] = pos_enc
        elif self.use_features:
            for i in range(self.featured_level_num): # from finest to coarsest
                cur_lvl = self.max_level - i
                fea_lvl = self.featured_level_num - i - 1
                xy_indices = indices[i][0]; xy_fea_cor = self.xy_hier_features[fea_lvl][xy_indices]
                xz_indices = indices[i][1]; xz_fea_cor = self.xz_hier_features[fea_lvl][xz_indices]
                yz_indices = indices[i][2]; yz_fea_cor = self.yz_hier_features[fea_lvl][yz_indices]
                xy_coeffts = self.interpolat(x, cur_lvl, 'xy')
                xz_coeffts = self.interpolat(x, cur_lvl, 'xz')
                yz_coeffts = self.interpolat(x, cur_lvl, 'yz')
                xy_fea = torch.einsum('ijk,ij->ik', xy_fea_cor, xy_coeffts)
                xz_fea = torch.einsum('ijk,ij->ik', xz_fea_cor, xz_coeffts)
                yz_fea = torch.einsum('ijk,ij->ik', yz_fea_cor, yz_coeffts)

                # Set masks for free free space points
                xy_masks = xy_indices >= 0; xy_masks_c = torch.all(xy_masks, dim=1)
                xz_masks = xz_indices >= 0; xz_masks_c = torch.all(xz_masks, dim=1)
                yz_masks = yz_indices >= 0; yz_masks_c = torch.all(yz_masks, dim=1)
                # all `true` means valid points
                masks = xy_masks_c & xz_masks_c & yz_masks_c
                _masks = torch.logical_not(masks)
                lvl_fea = xy_fea + xz_fea + yz_fea
                # the invalid points are set to zero.
                lvl_fea[_masks] = 0.
                sum_features[:, fea_lvl*self.feature_dim:(fea_lvl+1)*self.feature_dim] = lvl_fea
        return sum_features


    def query_feature(self,x, faster = False):
        # TODO how to extract indices in a fast way?
        self.set_zero()
        indices = self.get_indices(x)
        features = self.query_fea_by_indices(x, indices)
        return features

    def update(self, surface_points, incremental_on = False, origin = None):
        """
        surface_points: (N, 3) tensor
        """
        spc = kal.ops.conversions.unbatched_pointcloud_to_spc(surface_points, self.max_level, surface_points)
        pyramid = spc.pyramids[0].cpu()
        for i in range(self.max_level+1): # for each level (top-down)
            if i < self.free_level_num: # free levels (skip), only need to consider the featured levels
                continue
            nodes = spc.point_hierarchies[pyramid[1, i]:pyramid[1, i+1]] # The nodes are distributed in the [0,1] cube, meaning the bottom_left corner

            node_morton = kal.ops.spc.points_to_morton(nodes).cpu().numpy().tolist()
            new_node_idx = []
            for idx in range(len(node_morton)):
                if node_morton[idx] not in self.nodes_lookup_tables[i]:
                    new_node_idx.append(idx)
                    # TODO from the node lookup table directly get the feature plane indices
                    self.nodes_lookup_tables[i][node_morton[idx]] = True
            new_nodes = nodes[new_node_idx]
            if new_nodes.shape[0] == 0:
                continue

            'xy-plane'
            nodes_xy = copy.deepcopy(new_nodes); nodes_xy[:, 2] = 0; nodes_xy_uni, _ = torch.unique(nodes_xy, dim=0, return_inverse=True)
            nodes_xy_nui_m = kal.ops.spc.points_to_morton(nodes_xy_uni).cpu().numpy().tolist()
            new_node_xy_idx = []
            for idx in range(len(nodes_xy_nui_m)):
                if nodes_xy_nui_m[idx] not in self.xy_nodes_lookup_tables[i]:
                    new_node_xy_idx.append(idx)
            new_xy_nodes = nodes_xy_uni[new_node_xy_idx] # The real new nodes on xy plane
            if new_xy_nodes.shape[0] != 0:
                new_xy_cubic_corners = kal.ops.spc.points_to_corners(new_xy_nodes).reshape(-1,3)
                xy_corners = new_xy_cubic_corners[new_xy_cubic_corners[:, 2] == 0] # all xy plane corners attached to the new xy nodes

                xy_corners_unique = torch.unique(xy_corners, dim=0)
                xy_cor_uni_m = kal.ops.spc.points_to_morton(xy_corners_unique).cpu().numpy().tolist()
                if len(self.xy_corners_lookup_tables[i]) == 0:
                    self.xy_corners_lookup_tables[i] = dict(zip(xy_cor_uni_m, range(len(xy_cor_uni_m))))
                    fts = self.feature_std * torch.randn(len(xy_corners_unique) + 1, self.feature_dim).to(self.device)
                    fts[-1] = torch.zeros(1,self.feature_dim)
                    self.xy_hier_features.append(nn.Parameter(fts))
                else: # add new corners
                    new_xy_corner_idx = []
                    for idx, m in enumerate(xy_cor_uni_m):
                        if m not in self.xy_corners_lookup_tables[i]:
                            self.xy_corners_lookup_tables[i][m] = len(self.xy_corners_lookup_tables[i])
                            new_xy_corner_idx.append(idx)
                    new_xy_fts_num = len(new_xy_corner_idx)
                    new_xy_fts = self.feature_std * torch.randn(new_xy_fts_num+1, self.feature_dim).to(self.device)
                    new_xy_fts[-1] = torch.zeros(1,self.feature_dim)
                    cur_fea_lvl = i - self.free_level_num
                    # TODO test the number
                    self.xy_hier_features[cur_fea_lvl] = nn.Parameter(torch.cat([self.xy_hier_features[cur_fea_lvl][:-1], new_xy_fts], dim=0))
                # update the node lookup table
                xy_corners_m = kal.ops.spc.points_to_morton(xy_corners).cpu().numpy().tolist()
                xy_indices = torch.tensor([self.xy_corners_lookup_tables[i][m] for m in xy_corners_m]).reshape(-1,4).numpy().tolist()
                new_xy_nodes_m = kal.ops.spc.points_to_morton(new_xy_nodes).cpu().numpy().tolist()
                for k in range(len(new_xy_nodes_m)):
                    self.xy_nodes_lookup_tables[i][new_xy_nodes_m[k]] = xy_indices[k]

            'xz-plane'
            nodes_xz = copy.deepcopy(new_nodes); nodes_xz[:, 1] = 0; nodes_xz_uni, _ = torch.unique(nodes_xz, dim=0, return_inverse=True)
            nodes_xz_nui_m = kal.ops.spc.points_to_morton(nodes_xz_uni).cpu().numpy().tolist()
            new_node_xz_idx = []
            for idx in range(len(nodes_xz_nui_m)):
                if nodes_xz_nui_m[idx] not in self.xz_nodes_lookup_tables[i]:
                    new_node_xz_idx.append(idx)
            new_xz_nodes = nodes_xz_uni[new_node_xz_idx] # The real new nodes on xz plane
            if new_xz_nodes.shape[0] != 0:
                new_xz_cubic_corners = kal.ops.spc.points_to_corners(new_xz_nodes).reshape(-1,3)
                xz_corners = new_xz_cubic_corners[new_xz_cubic_corners[:, 1] == 0]

                xz_corners_unique = torch.unique(xz_corners, dim=0)
                xz_cor_uni_m = kal.ops.spc.points_to_morton(xz_corners_unique).cpu().numpy().tolist()
                if len(self.xz_corners_lookup_tables[i]) == 0:
                    self.xz_corners_lookup_tables[i] = dict(zip(xz_cor_uni_m, range(len(xz_cor_uni_m))))
                    fts = self.feature_std * torch.randn(len(xz_corners_unique) + 1, self.feature_dim).to(self.device)
                    fts[-1] = torch.zeros(1,self.feature_dim)
                    self.xz_hier_features.append(nn.Parameter(fts))
                else: # add new corners
                    new_xz_corner_idx = []
                    for idx, m in enumerate(xz_cor_uni_m):
                        if m not in self.xz_corners_lookup_tables[i]:
                            self.xz_corners_lookup_tables[i][m] = len(self.xz_corners_lookup_tables[i])
                            new_xz_corner_idx.append(idx)
                    new_xz_fts_num = len(new_xz_corner_idx)
                    new_xz_fts = self.feature_std * torch.randn(new_xz_fts_num + 1, self.feature_dim).to(self.device)
                    new_xz_fts[-1] = torch.zeros(1,self.feature_dim)
                    cur_fea_lvl = i - self.free_level_num
                    self.xz_hier_features[cur_fea_lvl] = nn.Parameter(torch.cat([self.xz_hier_features[cur_fea_lvl][:-1], new_xz_fts], dim=0))
                # update the node lookup table
                xz_corners_m = kal.ops.spc.points_to_morton(xz_corners).cpu().numpy().tolist()
                xz_indices = torch.tensor([self.xz_corners_lookup_tables[i][m] for m in xz_corners_m]).reshape(-1,4).numpy().tolist()
                new_xz_nodes_m = kal.ops.spc.points_to_morton(new_xz_nodes).cpu().numpy().tolist()
                for k in range(len(new_xz_nodes_m)):
                    self.xz_nodes_lookup_tables[i][new_xz_nodes_m[k]] = xz_indices[k]

            'yz-plane'
            nodes_yz = copy.deepcopy(new_nodes); nodes_yz[:, 0] = 0; nodes_yz_uni, _ = torch.unique(nodes_yz, dim=0, return_inverse=True)
            nodes_yz_nui_m = kal.ops.spc.points_to_morton(nodes_yz_uni).cpu().numpy().tolist()
            new_node_yz_idx = []
            for idx in range(len(nodes_yz_nui_m)):
                if nodes_yz_nui_m[idx] not in self.yz_nodes_lookup_tables[i]:
                    new_node_yz_idx.append(idx)
            new_yz_nodes = nodes_yz_uni[new_node_yz_idx] # The real new nodes on yz plane
            if new_yz_nodes.shape[0] != 0:
                new_yz_cubic_corners = kal.ops.spc.points_to_corners(new_yz_nodes).reshape(-1,3)
                yz_corners = new_yz_cubic_corners[new_yz_cubic_corners[:, 0] == 0]

                yz_corners_unique = torch.unique(yz_corners, dim=0)
                yz_cor_uni_m = kal.ops.spc.points_to_morton(yz_corners_unique).cpu().numpy().tolist()
                if len(self.yz_corners_lookup_tables[i]) == 0:
                    self.yz_corners_lookup_tables[i] = dict(zip(yz_cor_uni_m, range(len(yz_cor_uni_m))))
                    fts = self.feature_std * torch.randn(len(yz_corners_unique) + 1, self.feature_dim).to(self.device)
                    fts[-1] = torch.zeros(1,self.feature_dim)
                    self.yz_hier_features.append(nn.Parameter(fts))
                else: # add new corners
                    new_yz_corner_idx = []
                    for idx, m in enumerate(yz_cor_uni_m):
                        if m not in self.yz_corners_lookup_tables[i]:
                            self.yz_corners_lookup_tables[i][m] = len(self.yz_corners_lookup_tables[i])
                            new_yz_corner_idx.append(idx)
                    new_yz_fts_num = len(new_yz_corner_idx)
                    new_yz_fts = self.feature_std * torch.randn(new_yz_fts_num + 1, self.feature_dim).to(self.device)
                    new_yz_fts[-1] = torch.zeros(1,self.feature_dim)
                    cur_fea_lvl = i - self.free_level_num
                    self.yz_hier_features[cur_fea_lvl] = nn.Parameter(torch.cat([self.yz_hier_features[cur_fea_lvl][:-1], new_yz_fts], dim=0))
                # update the node lookup table
                yz_corners_m = kal.ops.spc.points_to_morton(yz_corners).cpu().numpy().tolist()
                yz_indices = torch.tensor([self.yz_corners_lookup_tables[i][m] for m in yz_corners_m]).reshape(-1,4).numpy().tolist()
                new_yz_nodes_m = kal.ops.spc.points_to_morton(new_yz_nodes).cpu().numpy().tolist()
                for k in range(len(new_yz_nodes_m)):
                    self.yz_nodes_lookup_tables[i][new_yz_nodes_m[k]] = yz_indices[k]

    def cal_regularization_loss(self):
        loss = torch.tensor(0.).to(self.device)
        with torch.no_grad():
            if len(self.xy_hier_features_last)==0 or len(self.yz_hier_features_last)==0 or len(self.xz_hier_features_last)==0:
                # initialization
                self.xy_hier_features_last = copy.deepcopy(self.xy_hier_features)
                self.xz_hier_features_last = copy.deepcopy(self.xz_hier_features)
                self.yz_hier_features_last = copy.deepcopy(self.yz_hier_features)
            else:
                # calculate the regularization loss
                for i in range(len(self.xy_hier_features)):
                    loss += torch.sum(torch.square(self.xy_hier_features[i] - self.xy_hier_features_last[i]))
                    loss += torch.sum(torch.square(self.xz_hier_features[i] - self.xz_hier_features_last[i]))
                    loss += torch.sum(torch.square(self.yz_hier_features[i] - self.yz_hier_features_last[i]))
                # update the last features
                self.xy_hier_features_last = copy.deepcopy(self.xy_hier_features)
                self.xz_hier_features_last = copy.deepcopy(self.xz_hier_features)
                self.yz_hier_features_last = copy.deepcopy(self.yz_hier_features)
        return loss

    def get_octree_nodes(self, level): # top-down
        nodes_morton = list(self.nodes_lookup_tables[level].keys())
        nodes_morton = torch.tensor(nodes_morton).to(self.device, torch.int64)
        nodes_spc = kal.ops.spc.morton_to_points(nodes_morton)
        nodes_spc_np = nodes_spc.cpu().numpy()
        node_size = 2**(1-level) # in the -1 to 1 kaolin space
        nodes_coord_scaled = (nodes_spc_np * node_size) - 1. + 0.5 * node_size  # in the -1 to 1 kaolin space
        return nodes_coord_scaled
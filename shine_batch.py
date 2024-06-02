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


def run_mapping_batch():

    config = Config()
    if len(sys.argv) > 1:
        config.load(sys.argv[1])
    else:
        sys.exit(
            "No configuration file"
        )

    run_path = setup_experiment(config)
    dev = config.device

    fea_enc = FeatureEncoder(config).fea_encoder
    geo_mlp = Decoder(config, is_geo_encoder=True, is_time_conditioned=config.time_conditioned, in_dim=fea_enc.out_dim)
    sem_mlp = Decoder(config, is_geo_encoder=False)

    # load the decoder model
    """
    if config.load_model:
        loaded_model = torch.load(config.model_path)
        geo_mlp.load_state_dict(loaded_model["geo_decoder"])
        print("Pretrained decoder loaded")
        freeze_model(geo_mlp) # fixed the decoder
        if config.semantic_on:
            sem_mlp.load_state_dict(loaded_model["sem_decoder"])
            freeze_model(sem_mlp) # fixed the decoder
        if 'feature_octree' in loaded_model.keys(): # also load the feature octree
            octree = loaded_model["feature_octree"]
            octree.print_detail()
    """

    # dataset
    dataset = LiDARDataset(config, fea_enc)

    # NOTE only for debug now
    mesher = Mesher(config, fea_enc, geo_mlp, sem_mlp)
    mesher.global_transform = inv(dataset.begin_pose_inv)

    # Visualizer on
    if config.o3d_vis_on:
        vis = MapVisualizer()

    # for each frame
    print("Load, preprocess and sample data")
    frame_cnt = 0
    for frame_id in tqdm(range(dataset.total_pc_count)):
        """
        # For kitti, no need to load the last frame
        if frame_id == dataset.total_pc_count-1:
            print("force to load the last frame:", frame_id)
            dataset.process_frame(frame_id)
            break
        """
        if (frame_id < config.begin_frame or frame_id > config.end_frame or \
            frame_id % config.every_frame != 0):
            continue
        frame_cnt += 1
        if frame_cnt % config.save_freq_iters == 0:
            checkpoint_name = 'model/model_feames_' + str(frame_cnt)
            save_checkpoint(fea_enc, geo_mlp, run_path, checkpoint_name, frame_cnt)
        dataset.process_frame(frame_id)

    bbox = dataset.map_bbx
    if bbox.min_bound[0] < -config.world_size or bbox.min_bound[1] < -config.world_size \
        or bbox.max_bound[0] > config.world_size or bbox.max_bound[1] > config.world_size:
        print("The bounding box is out of the world size, please check the config file")
        raise ValueError

    # learnable parameters
    fea_enc_param = list(fea_enc.parameters())
    geo_mlp_param = list(geo_mlp.parameters())
    sem_mlp_param = list(sem_mlp.parameters())
    # learnable sigma for differentiable rendering
    sigma_size = torch.nn.Parameter(torch.ones(1, device=dev)*1.0)
    # fixed sigma for sdf prediction supervised with BCE loss
    sigma_sigmoid = config.logistic_gaussian_ratio*config.sigma_sigmoid_m*config.scale

    pc_map_path = run_path + '/map/pc_map_down.ply'
    dataset.write_merged_pc(pc_map_path)

    # initialize the optimizer
    opt = setup_optimizer(config, fea_enc_param, geo_mlp_param, sem_mlp_param, sigma_size)
    # get the size of the parameters
    print("Number of parameters: ", sum(p.numel() for p in fea_enc_param) + \
        sum(p.numel() for p in geo_mlp_param))
    # write the number of parameters to the file
    if not config.debug:
        param_file = open(run_path + '/param.txt', 'w')
        param_file.write(str(sum(p.numel() for p in fea_enc_param) + \
            sum(p.numel() for p in geo_mlp_param)))
        param_file.close()

    if config.fea_encoder_type == 'fea_octree':
        fea_enc.print_detail()

    # octree.print_detail()

    if config.normal_loss_on or config.ekional_loss_on or config.proj_correction_on or config.consistency_loss_on:
        require_gradient = True
    else:
        require_gradient = False

    # begin training
    print("Begin mapping")
    cur_base_lr = config.lr

    iter = 0
    for iter in tqdm(range(config.iters)):

        T0 = get_time()
        # learning rate decay
        step_lr_decay(opt, cur_base_lr, iter, config.lr_decay_step, config.lr_iters_reduce_ratio)

        # load batch data (avoid using dataloader because the data are already in gpu, memory vs speed)
        if config.ray_loss: # loss computed based on each ray
            coord, sample_depth, ray_depth, normal_label, sem_label, weight = dataset.get_batch()
        else: # loss computed based on each point sample
            coord, sdf_label, origin, ts, normal_label, sem_label, weight = dataset.get_batch()

        if require_gradient:
            coord.requires_grad_(True)
        T1 = get_time()
        feature = fea_enc.query_feature(coord)

        if config.progressive:
            fea_len = fea_enc.fea_dim
            pe_len = fea_enc.input_ch_pos
            # Optimize the features and PE in a progressive way (coarse-to-fine)
            steps = config.iters / 3; idx = np.ceil(iter / steps)
            if idx < 3:
                opt_fea_len = int(idx * fea_len / 3)
                opt_pe_len = int(idx * pe_len / 3)
                # detailed information is set to zero first
                feature[:, -pe_len+opt_pe_len:] = 0.0
                feature[:, opt_fea_len:fea_len] = 0.0

        pred = geo_mlp.sdf(feature)

        T3 = get_time()

        surface_mask = weight > 0

        # if config.normal_loss_on or config.ekional_loss_on:
        # use non-projective distance, gradually refined
        if require_gradient:
            g = get_gradient(coord, pred)*sigma_sigmoid

        if config.proj_correction_on:
            cos = torch.abs(F.cosine_similarity(g, coord - origin))
            cos[~surface_mask] = 1.0
            sdf_label = sdf_label * cos

        # consistency_loss_on is not set in the config file
        if config.consistency_loss_on:
            near_index = torch.randint(0, coord.shape[0], (min(config.consistency_count,coord.shape[0]),), device=dev)
            shift_scale = config.consistency_range * config.scale # 10 cm
            random_shift = torch.rand_like(coord) * 2 * shift_scale - shift_scale
            coord_near = coord + random_shift
            coord_near = coord_near[near_index, :] # only use a part of these coord to speed up
            coord_near.requires_grad_(True)
            feature_near = octree.query_feature(coord_near)
            pred_near = geo_mlp.sdf(feature_near)
            g_near = get_gradient(coord_near, pred_near)*sigma_sigmoid

        cur_loss = 0.
        # calculate the loss
        if config.ray_loss: # neural rendering loss
            pred_occ = torch.sigmoid(pred/sigma_size) # as occ. prob.
            pred_ray = pred_occ.reshape(config.bs, -1)
            sample_depth = sample_depth.reshape(config.bs, -1)
            if config.main_loss_type == "dr":
                dr_loss = batch_ray_rendering_loss(sample_depth, pred_ray, ray_depth, neus_on=False)
            elif config.main_loss_type == "dr_neus":
                dr_loss = batch_ray_rendering_loss(sample_depth, pred_ray, ray_depth, neus_on=True)
            cur_loss += dr_loss
        else: # sdf regression loss
            weight = torch.abs(weight) # weight's sign indicate the sample is around the surface or in the free space
            if config.main_loss_type == "sdf_bce": # our proposed method
                sdf_loss = sdf_bce_loss(pred, sdf_label, sigma_sigmoid, weight, config.loss_weight_on, config.loss_reduction)
            elif config.main_loss_type == "sdf_l1":
                sdf_loss = sdf_diff_loss(pred, sdf_label, weight, config.scale, l2_loss=False)
            elif config.main_loss_type == "sdf_l2":
                sdf_loss = sdf_diff_loss(pred, sdf_label, weight, config.scale, l2_loss=True)
            cur_loss += sdf_loss

        # optional loss (ekional, normal, gradient consistency loss)
        eikonal_loss = 0.
        if config.ekional_loss_on:
            eikonal_loss = ((1.0 - g[surface_mask].norm(2, dim=-1)) ** 2).mean() # MSE with regards to 1
            cur_loss += config.weight_e * eikonal_loss

        """
        consistency_loss = 0.
        if config.consistency_loss_on:
            consistency_loss = (1.0 - F.cosine_similarity(g[near_index, :], g_near)).mean()
            cur_loss += config.weight_c * consistency_loss
        """
        regularization_loss = 0.
        if config.regularization_loss_on:
            regularization_loss = fea_enc.cal_regularization_loss()
            cur_loss += config.weight_r * regularization_loss
        if config.record_feature_loss:
            regularization_loss = fea_enc.cal_regularization_loss() * config.weight_r

        normal_loss = 0.
        if config.normal_loss_on:
            g_direction = g / g.norm(2, dim=-1)
            normal_diff = g_direction - normal_label
            normal_loss = (normal_diff[surface_mask].abs()).norm(2, dim=1).mean()
            cur_loss += config.weight_n * normal_loss

        # semantic classification loss
        sem_loss = 0.
        if config.semantic_on:
            loss_nll = nn.NLLLoss(reduction='mean')
            sem_loss = loss_nll(sem_pred[::config.sem_label_decimation,:], sem_label[::config.sem_label_decimation])
            cur_loss += config.weight_s * sem_loss

        T4 = get_time()
        # save the loss and iter to the file
        if not config.debug:
            loss_file = open(run_path + '/loss.txt', 'a')
            if iter % 10 == 0:
                print('iter:', iter, 'loss:', cur_loss.item())
                if config.regularization_loss_on or config.record_feature_loss:
                    loss_file.write(str(iter) + ' ' + str(cur_loss.item()) + ' ' + str(sdf_loss.item()) + ' ' + str(regularization_loss.detach().cpu().numpy()) + '\n')
                else:
                    loss_file.write(str(iter) + ' ' + str(cur_loss.item()) + '\n')
        opt.zero_grad(set_to_none=True)
        cur_loss.backward()
        opt.step()

        T5 = get_time()

        # log to wandb
        if config.wandb_vis_on:
            if config.ray_loss:
                wandb_log_content = {'iter': iter, 'loss/total_loss': cur_loss, 'loss/render_loss': dr_loss, 'loss/eikonal_loss': eikonal_loss, 'loss/normal_loss': normal_loss, 'para/sigma': sigma_size}
            else:
                wandb_log_content = {'iter': iter, 'loss/total_loss': cur_loss, 'loss/sdf_loss': sdf_loss, 'loss/eikonal_loss': eikonal_loss, 'loss/normal_loss': normal_loss, 'loss/consistency_loss': consistency_loss, 'loss/sem_loss': sem_loss}
            wandb_log_content['timing(s)/load'] = T1 - T0
            wandb_log_content['timing(s)/get_indices'] = T2 - T1
            wandb_log_content['timing(s)/inference'] = T3 - T2
            wandb_log_content['timing(s)/cal_loss'] = T4 - T3
            wandb_log_content['timing(s)/back_prop'] = T5 - T4
            wandb_log_content['timing(s)/total'] = T5 - T0
            wandb.log(wandb_log_content)

        # reconstruction by marching cubes
        # if (((iter+1) % config.vis_freq_iters) == 0 and iter > 0):
        #     print("Begin mesh reconstruction from the implicit map")
        #     if not config.time_conditioned:
        #         mesh_path = run_path + '/mesh/mesh_iter_' + str(iter+1) + ".ply"
        #         map_path = run_path + '/map/sdf_map_iter_' + str(iter+1) + ".ply"
        #         if config.mc_with_octree: # default
        #             cur_mesh = mesher.recon_octree_mesh(config.mc_query_level, config.mc_res_m, mesh_path, map_path, config.save_map, config.semantic_on)
        #         else:
        #             cur_mesh = mesher.recon_bbx_mesh(dataset.map_bbx, config.mc_res_m, mesh_path, map_path, config.save_map, config.semantic_on)
        #         if config.o3d_vis_on:
        #             cur_mesh.transform(dataset.begin_pose_inv)
        #             vis.update_mesh(cur_mesh)
        #     else:
        #         vis.stop()
        #         for frame_id in tqdm(range(dataset.total_pc_count)):
        #             if (frame_id < config.begin_frame or frame_id > config.end_frame or \
        #                 frame_id % 2 != 0):
        #                 continue
        #             mesher.ts = frame_id
        #             mesh_path = run_path + '/mesh/mesh_iter_' + str(iter+1) + '_ts_' + str(frame_id) + ".ply"
        #             map_path = run_path + '/map/sdf_map_iter_' + str(iter+1) + '_ts_' + str(frame_id) + ".ply"
        #             if config.mc_with_octree: # default
        #                 cur_mesh = mesher.recon_octree_mesh(config.mc_query_level, config.mc_res_m, mesh_path, map_path, config.save_map, config.semantic_on)
        #             else:
        #                 cur_mesh = mesher.recon_bbx_mesh(dataset.map_bbx, config.mc_res_m, mesh_path, map_path, config.save_map, config.semantic_on)

        #             if config.o3d_vis_on:
        #                 cur_mesh.transform(dataset.begin_pose_inv)
        #                 vis.update_mesh(cur_mesh)
    # at last, always reconstruct the mesh

    # save checkpoint model
    checkpoint_name = 'model/model_iter_' + str(iter+1)
    # octree.clear_temp()
    save_checkpoint(fea_enc, geo_mlp, run_path, checkpoint_name, iter)

    # copy the config file to the run path
    if not config.debug:
        copyfile(sys.argv[1], run_path + '/config.yaml')
    mesh_path = run_path + '/mesh/mesh_iter_' + str(iter+1) + ".ply"
    map_path = run_path + '/map/sdf_map_iter_' + str(iter+1) + '_ts_' + str(0) + ".ply"
    cur_mesh = mesher.recon_bbx_mesh(dataset.map_bbx, config.mc_res_m, mesh_path, map_path, config.save_map, config.semantic_on)
    # cur_mesh = mesher.recon_octree_mesh(config.mc_query_level, config.mc_res_m, mesh_path, map_path, config.save_map, config.semantic_on)

    if config.o3d_vis_on:
        vis.stop()

if __name__ == "__main__":
    run_mapping_batch()

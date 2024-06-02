import csv
from eval_utils import eval_mesh

########################################### MaiCity Dataset ###########################################
# dataset_name = "maicity_01_"

# for evaluating completeness
# gt_pcd_path = "/home/shuo/Downloads/gt_map_pc_mai.ply"

# for evaluating accuracy
# gt_pcd_path = "/media/shuo/T7/mai_city/mai_city/ply/sequences/02/gt/gt_map_pc_mai_inter_croped.ply"

# vbdfusion====================
    # dense inputs
# pred_mesh_dir = "/media/shuo/T7/experiment_result/mai_city/dense/vdbfusion/"
# pred_mesh_paths = [pred_mesh_dir + "mesh5.ply"]
# method_name = "vdbfusion_dense"

    # Sparse inputs
# pred_mesh_dir = "/media/shuo/T7/experiment_result/mai_city/sparse/vdbfusion/"
# steps = [6]
# pred_mesh_paths = [pred_mesh_dir + f"mesh_every_{step}.ply" for step in steps]
# method_name = [str("vdbfusion_sparse_every" + str(step) + "frame") for step in steps]

# shine-mapping====================
    # dense inputs
# method_name = "shine_mapping_dense_10cm"
# pred_mesh_dir = "/media/shuo/T7/experiment_result/mai_city/dense/shine_mapping/"
# pred_mesh_paths = [pred_mesh_dir + "shine_mapping_10cm_voxel.ply"]

    # sparse inputs
# pred_mesh_dir = "/media/shuo/T7/experiment_result/mai_city/sparse/shine_mapping/"
# steps = [6]
# pred_mesh_paths = [pred_mesh_dir + f"mesh_every_{step}.ply" for step in steps]
# method_name = [str("shine_mapping_sparse_every" + str(step) + "frame") for step in steps]

# ours====================
    # dense
# method_name = "quadTree_pe_dense_acc"
# method_name = "ours_dense_local"
# pred_mesh_dir = "/media/shuo/T7/experiment_result/mai_city/dense/ours/"
# pred_mesh_paths = [pred_mesh_dir + "mesh_iter_50000.ply"]
    # sparse
# pred_mesh_dir = "/media/shuo/T7/experiment_result/mai_city/sparse/ours/"
# steps = [6]
# pred_mesh_paths = [pred_mesh_dir + f"mesh_every_{step}.ply" for step in steps]
# method_name = [str("ours_sparse_every" + str(step) + "frame") for step in steps]

# abliation study
    # map quality
# pred_mesh_dir = "/media/shuo/T7/experiment_result/abaliation_study/"
# mesh_name = ["fea_plane_only.ply", "full_model.ply", "pos_enc_only.ply"]
# pred_mesh_paths = [pred_mesh_dir + name for name in mesh_name]
# method_name = [str("ablation_study_" + name) for name in mesh_name]

    # convergence
        # fea only
# pred_mesh_dir = "/media/shuo/T7/experiment_result/abaliation_study/convergence/fea_only/"
# iterations = [25,50,75,100,125,150]
# pred_mesh_paths = [pred_mesh_dir + f"mesh_iter_{iter}.ply" for iter in iterations]
# method_name = [str("fea_only_iters" + f"{iter}") for iter in iterations]
        # fea + pos
# pred_mesh_dir = "/media/shuo/T7/experiment_result/abaliation_study/convergence/full/"
# iterations = [25,50,75,100,125,150]
# pred_mesh_paths = [pred_mesh_dir + f"mesh_iter_{iter}.ply" for iter in iterations]
# method_name = [str("full_iters" + f"{iter}") for iter in iterations]

######################################## Newer College Dataset ########################################
dataset_name = "ncd_quad_"

gt_pcd_path = "/media/shuo/T7/NewerCollege/ncd_example/quad/ncd_quad_gt_pc.ply"

    # shine-mapping
# pred_mesh_dir = "/media/shuo/T7/experiment_result/ncd/shine_mapping/"
# steps = [3,6,9,12,15,18]
# pred_mesh_paths = [pred_mesh_dir + f"mesh_every_{step}.ply" for step in steps]
# method_name = [str("shinemapping_every_" + str(step) + "frame") for step in steps]

    # vdbfusion
pred_mesh_dir = "/media/shuo/T7/experiment_result/ncd/vdbfusion/"
steps = [3,6,9,12,15,18]
pred_mesh_paths = [pred_mesh_dir + f"mesh_every_{step}.ply" for step in steps]
method_name = [str("vdbfusion_" + str(step) + "frame") for step in steps]

#     # ours
# pred_mesh_dir = "/media/shuo/T7/experiment_result/ncd/ours/"
# steps = [3,6,9,12,15,18]
# pred_mesh_paths = [pred_mesh_dir + f"mesh_every_{step}.ply" for step in steps]
# method_name = [str("ours_every_" + str(step) + "frame") for step in steps]

######################################## Newer College Dataset ########################################

# evaluation results output file
# NOTE modify
base_output_folder = "./experiments/evaluation/ncd/vdbfusion/"

# evaluation parameters
if dataset_name == 'ncd_quad_':
    # For NCD
    down_sample_vox = 0.02
    dist_thre = 0.2
    truncation_dist_acc = 0.4
    truncation_dist_com = 2.0
elif dataset_name == 'maicity_01_':
    # For MaiCity
    down_sample_vox = 0.02
    dist_thre = 0.1
    truncation_dist_acc = 0.2
    truncation_dist_com = 2.0
else:
    print("Dataset name not recognized")
    exit()


# evaluation
i = 0
for pred_mesh_path in pred_mesh_paths:
    output_csv_path = base_output_folder + dataset_name + method_name[i] + f"_eval.csv"
    eval_metric = eval_mesh(pred_mesh_path, gt_pcd_path, down_sample_res=down_sample_vox, threshold=dist_thre,
                            truncation_acc = truncation_dist_acc, truncation_com = truncation_dist_com, gt_bbx_mask_on = True)

    print(eval_metric)

    evals = [eval_metric]

    csv_columns = ['MAE_accuracy (m)', 'MAE_completeness (m)', 'Chamfer_L1 (m)', 'Chamfer_L2 (m)', \
            'Precision [Accuracy] (%)', 'Recall [Completeness] (%)', 'F-score (%)', 'Spacing (m)', \
            'Inlier_threshold (m)', 'Outlier_truncation_acc (m)', 'Outlier_truncation_com (m)']
    try:
        with open(output_csv_path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in evals:
                writer.writerow(data)
    except IOError:
        print("I/O error")

    i += 1
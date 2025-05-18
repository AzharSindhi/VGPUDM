import os
import open3d
import numpy as np
import torch
import shutil
from shutil import copyfile
import copy
import time
import argparse
import json
from dataset import get_dataloader
from util import calc_diffusion_hyperparams, set_seed
from models.pointnet2_with_pcld_condition import PointNet2CloudCondition

from util import sampling, sampling_ddim,calc_diffusion_hyperparams, AverageMeter,numpy_to_pc
from json_reader import replace_list_with_string_in_a_dict, restore_string_to_list_in_a_dict

import sys
sys.path.append(os.path.abspath("../"))
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist,hausdorff_distance


def evaluate(
        net,
        testloader,
        diffusion_hyperparams,
        print_every_n_steps=200,
        scale=1,
        compute_cd=True,
        return_all_metrics=False,
        R=4,
        npoints=2048,
        gamma=0.5,
        T=1000,
        step=30,
        mesh_path = "/mnt/SG10T/DataSet/PUGAN/test/mesh",
        p2f_root="../evaluation_code",
        save_dir = "./test/xys",
        save_xyz = True,            # pre dense point cloud
        save_sp=True,               # pre sparse point cloud
        save_z = False,             # input Gaussian noise
        save_condition = False,     # input sparse point cloud
        save_gt = False,            # true dense point cloud
        save_mesh = False,
        p2f = False,
):
    CD_meter = AverageMeter()
    HD_meter = AverageMeter()
    P2F_meter = AverageMeter()
    total_len = len(testloader)

    total_meta = torch.rand(0).cuda().long()

    metrics = {
        'cd_distance': torch.rand(0).cuda(),
        'h_distance': torch.rand(0).cuda(),
        'cd_p': torch.rand(0).cuda(),
    }

    cd_module = chamfer_3DDist()

    total_time = 0
    cd_result = 0
    times = 0
    mesh_path = mesh_path
    save_path = save_dir
    p2f_root = p2f_root
    save_xyz = save_xyz
    save_z = save_z
    save_condition =save_condition
    save_gt =save_gt
    save_sp = save_sp
    save_mesh = save_mesh
    p2f = p2f
    print(f"**** {npoints} -----> {npoints * R} ****")
    for idx, data in enumerate(testloader):

        label = data['label'].cuda()
        condition = data['partial'].cuda()
        gt = data['complete'].cuda()

        batch,num_points,_ = gt.shape
        net.reset_cond_features()
        start = time.time()

        start_time = time.time()

        if (step < T):
            generated_data, condition_pre, z = sampling_ddim(
                net=net,
                size=(batch, num_points, 3),
                diffusion_hyperparams=diffusion_hyperparams,
                label=label,
                condition=condition,
                R=R,
                gamma=gamma,
                step=step
            )
        else:
            generated_data,condition_pre,z = sampling(
                net=net,
                size=(batch,num_points,3),
                diffusion_hyperparams=diffusion_hyperparams,
                print_every_n_steps=print_every_n_steps,
                label=label,
                condition=condition,
                R=R,
                gamma=gamma
            )

        end_time = time.time() - start_time
        times += end_time

        generation_time = time.time() - start
        total_time = total_time + generation_time
        generated_data = generated_data/scale
        gt = gt/scale
        torch.cuda.empty_cache()

        if compute_cd:
            cd_p, dist, _,_ = cd_module(generated_data, gt)
            dist = (cd_p + dist) / 2.0
            cd_loss = dist.mean().detach().cpu().item()
        else:
            dist = torch.zeros(generated_data.shape[0], device=generated_data.device, dtype=generated_data.dtype)
            cd_p = dist
            cd_loss = dist.mean().detach().cpu().item()

        cd_result += torch.sum(cd_p).item()

        # ---- h distance ----
        hd_cost = hausdorff_distance(generated_data,gt)
        hd_loss = hd_cost.mean().detach().cpu().item()
        # ---- h distance ----

        # ---- p2f ----
        p2f_loss = 0
        names = data['name']
        if(p2f):
            global_p2f = []
            for name in names:
                p2f_path = os.path.join(p2f_root,f"{name}_point2mesh_distance.xyz")
                if(os.path.exists(p2f_path)):
                    point2mesh_distance = np.loadtxt(p2f_path).astype(np.float32)
                    if point2mesh_distance.size == 0:
                        continue
                    point2mesh_distance = point2mesh_distance[:, 3]
                    global_p2f.append(point2mesh_distance)
            global_p2f = np.concatenate(global_p2f, axis=0)
            p2f_loss = np.nanmean(global_p2f)
            p2f_std = np.nanstd(global_p2f)
        # ---- p2f ----
        total_meta = torch.cat([total_meta, label])

        metrics['cd_distance'] = torch.cat([metrics['cd_distance'], dist])
        metrics['h_distance'] = torch.cat([metrics['h_distance'], hd_cost])
        metrics['cd_p'] = torch.cat([metrics['cd_p'], cd_p])

        CD_meter.update(cd_loss, n=batch)
        HD_meter.update(hd_loss, n=batch)
        P2F_meter.update(p2f_loss, n=batch)

        print('progress [%d/%d] %.4f (%d samples) CD distance %.8f Hausdorff distance %.8f p2f %.8f this batch time %.2f total generation time %.2f' % (
            idx, total_len,
            idx/total_len,
            batch,
            CD_meter.avg,
            HD_meter.avg,
            P2F_meter.avg,
            generation_time,
            total_time
        ), flush=True)


        if(save_xyz):
            # ---- save data ----
            generated_np = generated_data.detach().cpu().numpy()
            if condition_pre is not None:
                condition_pre_np = condition_pre.detach().cpu().numpy()
            z_np = z.detach().cpu().numpy()
            gt_np = gt.detach().cpu().numpy()
            condition_np = condition.detach().cpu().numpy()
            for i in range(len(generated_np)):
                name = names[i]
                #name = idx
                # ---- generated ----
                generated_points = generated_np[i]
                generated_pc = numpy_to_pc(generated_points)
                generated_path = os.path.join(save_path,f"{name}.xyz")
                open3d.io.write_point_cloud(filename=generated_path,pointcloud=generated_pc)
                print(f"---- saving generated dense point cloud: {generated_path} ----")
                # ---- generated ----

                # ---- mesh ----
                if (save_mesh):
                    mesh_source = os.path.join(mesh_path,f"{name}.off")
                    mesh_dist = os.path.join(save_path,f"{name}.off")
                    shutil.copy(mesh_source,mesh_dist)
                    print(f"---- saving mesh: {mesh_dist} ----")
                # ---- mesh ----

                # ---- pre condition ----
                if(save_sp):
                    condition_pre_points = condition_pre_np[i]
                    condition_pre_pc = numpy_to_pc(condition_pre_points)
                    condition_pre_path = os.path.join(save_path,f"{name}_sp.xyz")
                    open3d.io.write_point_cloud(filename=condition_pre_path,pointcloud=condition_pre_pc)
                    print(f"---- saving generated sparse point cloud: {condition_pre_path} ----")
                # ---- pre condition ----

                # ---- z ----
                if(save_z):
                    z_points = z_np[i]
                    z_pc = numpy_to_pc(z_points)
                    z_path = os.path.join(save_path,f"{name}_z.xyz")
                    open3d.io.write_point_cloud(filename=z_path,pointcloud=z_pc)
                    print(f"---- saving input Gaussian noise: {z_path} ----")
                # ---- z ----

                # ---- gt ----
                if(save_gt):
                    gt_points = gt_np[i]
                    gt_pc = numpy_to_pc(gt_points)
                    gt_path = os.path.join(save_path,f"{name}_gt.xyz")
                    open3d.io.write_point_cloud(filename=gt_path,pointcloud=gt_pc)
                    print(f"---- saving truth dense point cloud: {gt_path} ----")
                # ---- gt ----

                # ---- condition ----
                if(save_condition):
                    condition_points = condition_np[i]
                    condition_pc = numpy_to_pc(condition_points)
                    condition_path = os.path.join(save_path,f"{name}_condition.xyz")
                    open3d.io.write_point_cloud(filename=condition_path,pointcloud=condition_pc)
                    print(f"---- saving input sparse point cloud: {condition_path} ----")
                # ---- condition ----
            # ---- save data ----

    total_meta = total_meta.detach().cpu().numpy()
    print(f"Times : {times}")
    if return_all_metrics:
        return CD_meter.avg, HD_meter.avg, P2F_meter.avg, total_meta, metrics
    else:
        return CD_meter.avg, HD_meter.avg, P2F_meter.avg, total_meta, metrics['cd_distance']


if __name__ == "__main__":
    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('-a', '--alpha', type=float, default=0.4)
    parser.add_argument('-g', '--gamma', type=float, default=0.5)
    parser.add_argument('-m', '--model_path', type=str, default="")
    parser.add_argument('-i', '--image_fusion_strategy', type=str, default="only_clip")
    parser.add_argument('-b', '--image_backbone', type=str, default="dino")
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--eval_batch_size', type=int, default=30)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    assert os.path.exists(args.model_path), f"Pretrained model path {args.model_path} does not exist"
    assert args.image_fusion_strategy in ['none', 'input', 'condition', 'second_condition', 'latent', 'only_clip', 'cross_attention'], f"Invalid image fusion strategy: {args.image_fusion_strategy}"
    assert args.image_backbone in ['clip', 'dino', 'none'], f"Invalid image backbone: {args.image_backbone}"
    args.run_dir = os.path.dirname(args.model_path)
    
    args.config = f"./exp_configs/{args.dataset}.json"
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    config = restore_string_to_list_in_a_dict(config)

    print('The configuration is:')
    print(json.dumps(replace_list_with_string_in_a_dict(copy.deepcopy(config)), indent=4))

    global train_config
    global pointnet_config
    global diffusion_config
    global trainset_config
    global diffusion_hyperparams

    train_config = config["train_config"]
    pointnet_config = config["pointnet_config"]
    diffusion_config = config["diffusion_config"]

    if train_config['dataset'] == 'PU1K':
        trainset_config = config["pu1k_dataset_config"]
    elif train_config['dataset'] == 'PUGAN':
        trainset_config = config['pugan_dataset_config']
    elif train_config['dataset'] == 'ViPC':
        trainset_config = config['vipc_dataset_config']
    elif train_config['dataset'] == 'ModelNet10':
        trainset_config = config['modelnet10_dataset_config']
    else:
        raise Exception('%s dataset is not supported' % train_config['dataset'])

    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)
    trainset_config["data_dir"] = os.path.expanduser(trainset_config["data_dir"])

    # override config from args
    pointnet_config['image_fusion_strategy'] = args.image_fusion_strategy
    pointnet_config['image_backbone'] = args.image_backbone
    trainset_config['batch_size'] = args.batch_size
    trainset_config['eval_batch_size'] = args.batch_size # equal to batch size
    trainset_config['debug'] = args.debug

    vis_dir = os.path.join(args.run_dir, "vis")
    # config_path = os.path.join(args.run_dir, os.path.split(config)[1])
    os.makedirs(vis_dir, exist_ok=True)
    # try:
    #     copyfile(config, config_path)
    # except:
    #     print('The two files are the same, no need to copy')

    print("vis directory is", vis_dir, flush=True)

    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    vis_dataloader = get_dataloader(trainset_config, phase="vis")
    net = PointNet2CloudCondition(pointnet_config).cuda()

    # ignore missing keys while loading checkpoints
    checkpoint = torch.load(args.model_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in net.state_dict()}
    net.load_state_dict(filtered_state_dict, strict=False)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # start_epoch = checkpoint.get('epoch', 0)
    # n_epochs = start_epoch + n_epochs
    # best_test_loss = checkpoint.get('best_test_loss', float('inf'))
    print(f'Loaded checkpoint from {args.model_path}', flush=True)
    net.eval()

    ## -- run evaluation -- #####

    cd_meter_avg, hd_meter_avg, p2f_meter_avg, total_meta, _ = evaluate(
        net,
        vis_dataloader,
        diffusion_hyperparams,
        print_every_n_steps=200,
        scale=1,
        compute_cd=True,
        return_all_metrics=False,
        R=trainset_config["R"],
        npoints=trainset_config["npoints"],
        gamma=pointnet_config["gamma"],
        T=diffusion_config["T"],
        step=30,
        mesh_path = None,
        p2f_root=None,
        save_dir = vis_dir,
        save_xyz = True,            # pre dense point cloud
        save_sp=True,               # pre sparse point cloud
        save_z = False,             # input Gaussian noise
        save_condition = True,     # input sparse point cloud
        save_gt = True,            # true dense point cloud
        save_mesh = False,
        p2f = False,
    )

    print("cd_meter_avg: ", cd_meter_avg, flush=True)
    print("hd_meter_avg: ", hd_meter_avg, flush=True)




import os
import argparse
import json
import torch
from dataset import get_dataloader
from util import calc_diffusion_hyperparams, set_seed
from models.pointnet2_with_pcld_condition import PointNet2CloudCondition
from shutil import copyfile
import copy
from json_reader import replace_list_with_string_in_a_dict, restore_string_to_list_in_a_dict
import sys
sys.path.append(os.path.abspath("../"))
from eval import evaluate as generate_samples


def evaluate(config_file, model_path, out_dir):
    # create the tensorboard run with run_name
    vis_dir = os.path.join(out_dir, "vis")
    config_path = os.path.join(out_dir, os.path.split(config_file)[1])
    os.makedirs(vis_dir, exist_ok=True)
    try:
        copyfile(config_file, config_path)
    except:
        print('The two files are the same, no need to copy')

    print("vis directory is", vis_dir, flush=True)
    print("Config file has been copied from %s to %s" % (config_file, config_path), flush=True)

    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    vis_dataloader = get_dataloader(trainset_config, phase='vis')
    net = PointNet2CloudCondition(pointnet_config).cuda()

    # ignore missing keys while loading checkpoints
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in net.state_dict()}
    net.load_state_dict(filtered_state_dict, strict=False)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # start_epoch = checkpoint.get('epoch', 0)
    # n_epochs = start_epoch + n_epochs
    # best_test_loss = checkpoint.get('best_test_loss', float('inf'))
    print(f'Loaded checkpoint from {model_path}', flush=True)


    cd_meter_avg, hd_meter_avg, p2f_meter_avg, total_meta, _ = generate_samples(
        net,
        vis_dataloader,
        diffusion_hyperparams,
        print_every_n_steps=200,
        scale=1,
        compute_cd=True,
        return_all_metrics=False,
        R=pointnet_config["R"],
        npoints=pointnet_config["npoints"],
        gamma=pointnet_config["gamma"],
        T=diffusion_config["T"],
        step=30,
        mesh_path = None,
        p2f_root=None,
        save_dir = vis_dir,
        save_xyz = False,            # pre dense point cloud
        save_sp=False,               # pre sparse point cloud
        save_z = False,             # input Gaussian noise
        save_condition = False,     # input sparse point cloud
        save_gt = False,            # true dense point cloud
        save_mesh = False,
        p2f = False,
    )

    print("cd_meter_avg: ", cd_meter_avg, flush=True)
    print("hd_meter_avg: ", hd_meter_avg, flush=True)

if __name__ == "__main__":
    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('-a', '--alpha', type=float, default=1.0)
    parser.add_argument('-g', '--gamma', type=float, default=0.5)
    parser.add_argument('-m', '--model_path', type=str, default="")
    parser.add_argument('-i', '--image_fusion_strategy', type=str, default="only_clip")
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--early_stopping_patience', type=int, default=100)
    parser.add_argument('--run_name', type=str, default="")
    parser.add_argument('--image_backbone', type=str, default="dino")
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--eval_batch_size', type=int, default=40)
    args = parser.parse_args()

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
    run_name = f"{args.image_backbone}_{args.dataset}_{args.image_fusion_strategy}"
    trainset_config["data_dir"] = os.path.expanduser(trainset_config["data_dir"])
    assert args.image_fusion_strategy in ['none', 'input', 'condition', 'second_condition', 'latent', 'only_clip'], f"Invalid image fusion strategy: {args.image_fusion_strategy}"
    assert args.image_backbone in ['clip', 'dino', 'none'], f"Invalid image backbone: {args.image_backbone}"
    if args.model_path != "":
        assert os.path.exists(args.model_path), f"Pretrained model path {args.model_path} does not exist"
        run_name += "_pre" + os.path.basename(args.model_path).split(".")[0]
    if args.debug:
        run_name += "_debug"

    # override config from args
    pointnet_config['image_fusion_strategy'] = args.image_fusion_strategy
    pointnet_config['image_backbone'] = args.image_backbone
    trainset_config['batch_size'] = args.batch_size
    trainset_config['eval_batch_size'] = args.batch_size # equal to batch size
    trainset_config['debug'] = args.debug

    train(
        args.config,
        args.model_path,
        **train_config,
        run_name=run_name,
    )

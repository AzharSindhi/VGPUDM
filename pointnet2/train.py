import os
import time
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from termcolor import cprint
from dataset import get_dataloader
from util import training_loss, calc_diffusion_hyperparams, find_max_epoch, print_size, set_seed
from models.pointnet2_with_pcld_condition import PointNet2CloudCondition
from shutil import copyfile
import copy
from json_reader import replace_list_with_string_in_a_dict, restore_string_to_list_in_a_dict
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
import sys
sys.path.append(os.path.abspath("../"))
from eval import evaluate as generate_samples
from contextlib import redirect_stdout
import io
from torch.optim.lr_scheduler import ReduceLROnPlateau

def split_data(data):
    label = data['label'].cuda()
    X = data['complete'].cuda()
    condition = data['partial'].cuda()
    class_index = data['class_index'].cuda()
    return X, condition, class_index, label

def evaluate(model, testloader, diffusion_hyperparams, use_interpolation, progress, task_id):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(testloader):
            X, condition, class_index, label = split_data(data)
            loss = training_loss(model, nn.MSELoss(), X, diffusion_hyperparams, label=label, class_index=class_index, 
                                    condition=condition, use_interpolation=use_interpolation)
            total_loss += loss.item()
            avg_loss = total_loss / (i + 1)
            progress.update(task_id, advance=1, description=f"[Eval] Loss: {avg_loss:.4f}")
    return total_loss / len(testloader)

def train_one_epoch(model, optimizer, dataloader, diffusion_hyperparams, use_interpolation, progress, task_id):
    model.train()
    epoch_loss = 0.0
    for i, data in enumerate(dataloader):
        X, condition, class_index, label = split_data(data)
        loss = training_loss(model, nn.MSELoss(), X, diffusion_hyperparams, label=label, class_index=class_index,
                                    condition=condition, use_interpolation=use_interpolation)
        optimizer.zero_grad()
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        avg_loss = epoch_loss / (i + 1)
        progress.update(task_id, advance=1, description=f"[Train] Loss: {avg_loss:.4f}")
    return epoch_loss / len(dataloader)

def get_early_stopping_counter(test_loss, best_loss, previous_counter):
    if test_loss < best_loss:
        return 0
    else:
        return previous_counter + 1

def run_samples(net, vis_dataloader, vis_dir, use_interpolation):
    net.eval()
    with redirect_stdout(io.StringIO()):    
        cd_meter_avg, hd_meter_avg, p2f_meter_avg, total_meta, _ = generate_samples(
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
            step=diffusion_config["T"],
            mesh_path = None,
            p2f_root=None,
            save_dir = vis_dir,
            save_xyz = True,            # pre dense point cloud
            save_sp=False,               # pre sparse point cloud
            save_z = False,             # input Gaussian noise
            save_condition = True,     # input sparse point cloud
            save_gt = True,            # true dense point cloud
            save_mesh = False,
            p2f = False,
            use_interpolation = use_interpolation,
        )

        return cd_meter_avg, hd_meter_avg


def train(config_file, model_path, dataset, root_directory, run_name, n_epochs, epochs_per_ckpt, learning_rate):
    # create the tensorboard run with run_name
    tb = SummaryWriter(comment=run_name)
    output_directory = os.path.join(root_directory, run_name)
    vis_dir = os.path.join(root_directory, run_name, "vis")
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    try:
        copyfile(config_file, os.path.join(output_directory, os.path.split(config_file)[1]))
    except:
        print('The two files are the same, no need to copy')

    print("output directory is", output_directory, flush=True)
    print("Config file has been copied from %s to %s" % (config_file, os.path.join(output_directory, os.path.split(config_file)[1])), flush=True)

    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    trainloader = get_dataloader(trainset_config)
    testloader = get_dataloader(trainset_config, phase='test')
    vis_dataloader = get_dataloader(trainset_config, phase='train')
    net = PointNet2CloudCondition(pointnet_config).cuda()
    start_epoch = 0
    best_test_loss = float('inf')
    early_stopping_counter = 0
    cd_meter_avg = float('inf')
    hd_meter_avg = float('inf')
    test_loss = float('inf') # to log the first checkpoint
    test_log_interval = args.test_log_interval
    use_interpolation = pointnet_config['use_interpolation']

    if model_path != "":
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

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    console = Console()
    console.print(f"[bold yellow]Training for {n_epochs} epochs...[/bold yellow]")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        for epoch in range(start_epoch, n_epochs):
            console.print(f"\n[bold cyan]Epoch {epoch + 1}/{n_epochs}[/bold cyan]")
            epoch_start_time = time.time()
            train_task = progress.add_task("[Train]", total=len(trainloader))
            train_loss = train_one_epoch(net, optimizer, trainloader, diffusion_hyperparams, use_interpolation, progress, train_task)
            test_task = progress.add_task("[Eval]", total=len(testloader))
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            
            if epoch % test_log_interval == 0:
                test_loss = evaluate(net, testloader, diffusion_hyperparams, use_interpolation, progress, test_task)
                # if test_loss < best_test_loss:
                #     best_test_loss = test_loss
                #     early_stopping_counter = 0
                #     torch.save(checkpoint, os.path.join(output_directory, 'best_checkpoint.pt'))
                # else:
                #     early_stopping_counter += 1

                cd_meter_avg, hd_meter_avg = run_samples(net, vis_dataloader, vis_dir, args.use_interpolation)
                tb.add_scalar('CD_best', cd_meter_avg, epoch)
                tb.add_scalar('HD_best', hd_meter_avg, epoch)
                console.print(f"[yellow]CD_best: {cd_meter_avg:.4f} | HD_best: {hd_meter_avg:.4f}[/yellow]")
                
                # scheduler.step(test_loss)
            
            epoch_duration = time.time() - epoch_start_time
            torch.save(checkpoint, os.path.join(output_directory, 'latest_checkpoint.pt'))
            
            console.print(f"[green]Epoch Duration: {epoch_duration:.2f}s | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | CD_best: {cd_meter_avg:.4f} | HD_best: {hd_meter_avg:.4f}[/green]")
            tb.add_scalar('Log-Train-Loss', train_loss, epoch)
            tb.add_scalar('Log-Test-Loss', test_loss, epoch)

            
            progress.remove_task(train_task)
            progress.remove_task(test_task)
            if early_stopping_counter >= args.early_stopping_patience:
                console.print("[red]Early stopping triggered. Training stopped.[/red]")
                break
        
    tb.close()

if __name__ == "__main__":
    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default="ViPC")
    parser.add_argument('-a', '--alpha', type=float, default=1.0)
    parser.add_argument('-g', '--gamma', type=float, default=0.5)
    parser.add_argument('-m', '--model_path', type=str, default="")
    parser.add_argument('-i', '--image_fusion_strategy', type=str, default="only_clip")
    parser.add_argument('--no_cross_conditioning', action='store_true', default=False)
    parser.add_argument('--use_interpolation', action='store_true', default=True)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--mini', action='store_true', default=False)
    parser.add_argument('--early_stopping_patience', type=int, default=20)
    parser.add_argument('--run_name', type=str, default="")
    parser.add_argument('--image_backbone', type=str, default="clip")
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--eval_batch_size', type=int, default=30)
    parser.add_argument('--test_log_interval', type=int, default=20)
    parser.add_argument('--save_last_ckpt', action='store_true', default=False)
    args = parser.parse_args()

    args.config = f"./exp_configs/{args.dataset}.json"
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    config = restore_string_to_list_in_a_dict(config)

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
    run_name = f"{args.image_backbone}_{args.dataset}_{args.image_fusion_strategy}_cross_conditioning{not args.no_cross_conditioning}_rec{pointnet_config['condition_loss']}_sparseBranch{pointnet_config['include_local_feature']}_interpolation{args.use_interpolation}"
    trainset_config["data_dir"] = os.path.expanduser(trainset_config["data_dir"])
    assert args.image_fusion_strategy in ['none', 'input', 'condition', 'second_condition', 'latent', 'only_clip', 'cross_attention'], f"Invalid image fusion strategy: {args.image_fusion_strategy}"
    assert args.image_backbone in ['clip', 'dino', 'none'], f"Invalid image backbone: {args.image_backbone}"
    if args.model_path != "":
        assert os.path.exists(args.model_path), f"Pretrained model path {args.model_path} does not exist"
        run_name += "_pre" + os.path.basename(args.model_path).split(".")[0]

    # override config from args
    pointnet_config['image_fusion_strategy'] = args.image_fusion_strategy
    pointnet_config['use_cross_conditioning'] = not args.no_cross_conditioning
    pointnet_config['use_interpolation'] = args.use_interpolation
    pointnet_config['image_backbone'] = args.image_backbone
    trainset_config['batch_size'] = args.batch_size
    trainset_config['eval_batch_size'] = args.batch_size # equal to batch size
    trainset_config['debug'] = args.debug
    trainset_config['mini'] = args.mini
    if args.debug:
        run_name += "_debug"
    if args.mini:
        run_name += f"_mini_{trainset_config['category']}"
    
    run_name += f"_R{trainset_config['R']}"

    #TODO move config saving to the end
    print('The configuration is:')
    print(json.dumps(replace_list_with_string_in_a_dict(copy.deepcopy(config)), indent=4))

    train(
        args.config,
        args.model_path,
        **train_config,
        run_name=run_name,
    )

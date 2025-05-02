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
from util import training_loss, calc_diffusion_hyperparams,find_max_epoch, print_size,set_seed
from models.pointnet2_with_pcld_condition import PointNet2CloudCondition
from shutil import copyfile
import copy
from json_reader import replace_list_with_string_in_a_dict, restore_string_to_list_in_a_dict
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

def split_data(data):
    # load data
    label = data['label'].cuda()
    X = data['complete'].cuda()
    condition = data['partial'].cuda()
    class_index = data['class_index'].cuda()

    return X, condition, class_index, label

def evaluate(model, testloader, diffusion_hyperparams, progress, eval_task_id):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            # Assuming data format: [dense_pts, sparse_pts, class_index]
            X, condition, class_index, label = split_data(data)

            # Diffusion process and loss calculation
            loss = training_loss(model, nn.MSELoss(), X, diffusion_hyperparams, label=label, class_index=class_index, condition=condition)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(testloader)
    print(f'[Test] Test MSE Loss: {avg_loss:.4f}', flush=True)
    return avg_loss

def train_one_epoch(model, optimizer, dataloader, diffusion_hyperparams):
    """Runs the training loop for one epoch."""
    model.train()
    epoch_loss = 0.0
    total_steps_in_epoch = len(dataloader)

    for data in dataloader:
        # Assuming data format: [dense_pts, sparse_pts, class_index]
        X, condition, class_index, label = split_data(data)

        # Diffusion process and loss calculation
        loss = training_loss(model, nn.MSELoss(), X, diffusion_hyperparams, label=label, class_index=class_index, condition=condition)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / total_steps_in_epoch
    print(f'[Train] Train MSE Loss: {avg_train_loss:.4f}', flush=True)
    return avg_train_loss


def train(
        config_file,
        model_path,
        dataset,
        root_directory,
        output_directory,
        tensorboard_directory,
        n_epochs,
        epochs_per_ckpt,                # 当前多久保存一次模型
        learning_rate,
):

    local_path = dataset

    # Create tensorboard logger.
    tb = SummaryWriter(os.path.join(root_directory, local_path, tensorboard_directory))

    # Get shared output_directory ready
    output_directory = os.path.join(root_directory, local_path, output_directory)

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    try:
        copyfile(config_file, os.path.join(output_directory, os.path.split(config_file)[1]))
    except:
        print('The two files are the same, no need to copy')

    print("output directory is", output_directory, flush=True)
    print("Config file has been copied from %s to %s" % (config_file,
        os.path.join(output_directory, os.path.split(config_file)[1])), flush=True)

    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    trainloader = get_dataloader(trainset_config)

    # Config dataset and dataloader
    testloader = get_dataloader(trainset_config, phase='test')

    net = PointNet2CloudCondition(pointnet_config).cuda()
    net.train()

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # load checkpoint model
    start_epoch = 0
    best_test_loss = float('inf')
    # model_path = os.path.join(output_directory, 'latest_checkpoint.pt')
    # best_model_path = os.path.join(output_directory, 'best_checkpoint.pt')

    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] # Start from the next epoch
            best_test_loss = checkpoint.get('best_test_loss', float('inf')) # Load best loss if available
            print(f'Loaded checkpoint from {model_path}, resuming from epoch {start_epoch}', flush=True)
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.", flush=True)
            start_epoch = 0
            best_test_loss = float('inf')
    else:
        print("No valid checkpoint model found, start training from initialization.", flush=True)

    print(f"Training for {n_epochs} epochs...", flush=True)


    for epoch in range(start_epoch, n_epochs):
        epoch_start_time = time.time()
        
        # Train for one epoch
        train_loss = train_one_epoch(net, optimizer, trainloader, diffusion_hyperparams)
        test_loss = evaluate(net, testloader, diffusion_hyperparams)
        # log both losses here
        tb.add_scalar('Log-Train-Loss', train_loss, epoch)
        tb.add_scalar('Log-Test-Loss', test_loss, epoch)
        
        # Optional: Add epoch timing log here if needed
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch} duration: {epoch_duration:.2f}s", flush=True)

        # Checkpoint
        if (epoch) % epochs_per_ckpt == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            latest_ckpt_path = os.path.join(output_directory, 'latest_checkpoint.pt')
            torch.save(checkpoint, latest_ckpt_path)
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_ckpt_path = os.path.join(output_directory, 'best_checkpoint.pt')
                torch.save(checkpoint, best_ckpt_path)

    tb.close()


if __name__ == "__main__":

    # os.environ["CUDA_VISIBLE_DEVICES"]="1"


    # ---- config ----
    dataset="ModelNet10"
    check_name=""
    model_path = f"./exp_{dataset.lower()}/{dataset}/logs/checkpoint/{check_name}"
    alpha=1.0
    gamma=0.5
    set_seed(42)
    # ---- config ----

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default=dataset)
    parser.add_argument('-a', '--alpha', type=int, default=alpha)
    parser.add_argument('-g', '--gamma', type=int, default=gamma)
    parser.add_argument('-m', '--model_path', type=str, default=model_path)
    
    args = parser.parse_args()

    args.config = f"./exp_configs/{args.dataset}.json"
    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    config = restore_string_to_list_in_a_dict(config)
    print('The configuration is:')
    print(json.dumps(replace_list_with_string_in_a_dict(copy.deepcopy(config)), indent=4))

    # ---- global ----
    global train_config
    global pointnet_config
    global diffusion_config
    global trainset_config
    global diffusion_hyperparams
    # ---- global ----

    train_config = config["train_config"]        # training parameters
    pointnet_config = config["pointnet_config"]     # to define pointnet
    diffusion_config = config["diffusion_config"]    # basic hyperparameters
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
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)  # dictionary of all diffusion hyperparameters


    num_gpus = torch.cuda.device_count()

    train(
        args.config,
        args.model_path,
        **train_config,
    )

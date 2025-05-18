import torch
import torch.utils.data as data
from dataloader.dataset_loader import PUGAN,PU1K, ModelNet10
from dataloader.ViPCdataloader import  ViPCDataLoader

def get_dataloader(
        args,
        phase='train',
):
    if args['debug']:
        phase = 'test'
        augmentation = False
    if phase == 'vis':
        args['debug'] = True
        phase = "test"
        augmentation = False

    if phase == 'train':
        train = True
        shuffle = True
        batch_size = args['batch_size']
        augmentation = args.get('augmentation', False)
    else:
        assert phase in ['val', 'test', 'test_trainset']
        train = False
        shuffle = False
        batch_size = args['eval_batch_size']
        augmentation = False
        if phase == 'test_trainset':
            train = True

    return_augmentation_params = args.get('return_augmentation_params', False)
    if args.get('augment_data_during_generation', False):
        augmentation = args.get('augmentation', False)

    if args['dataset'] == 'PU1K':
        dataset = PU1K(
            args['data_dir'],
            train=train,
            scale=args['scale'],
            npoints=args['npoints'],
            augmentation=augmentation,
            return_augmentation_params=return_augmentation_params,
            R=args["R"],
            debug=args["debug"],
        )
        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=args['num_workers'],
            debug=args["debug"],
        )
    elif args['dataset'] == 'PUGAN':
        dataset = PUGAN(
            args['data_dir'],
            train=train,
            scale=args['scale'],
            npoints=args['npoints'],
            augmentation=augmentation,
            return_augmentation_params=return_augmentation_params,
            R=args["R"],
            debug=args["debug"],
        )
        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=args['num_workers'],
        )
    elif args['dataset'] == 'ViPC':
        phase = 'train' if train else 'test'
        dataset = ViPCDataLoader(
            args['data_dir'], phase, view_align=args['view_align'], 
            category=args['category'], mini=args['mini'],
            augmentation=augmentation,
            return_augmentation_params=return_augmentation_params,
            R=args["R"],
            debug=args["debug"],
            scale=args['scale'],
            image_size=args['image_size'],
        )
        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=args['num_workers'],
        )
    elif args['dataset'] == 'ModelNet10':
        dataset = ModelNet10(
            args['data_dir'],
            train=train,
            scale=args['scale'],
            npoints=args['npoints'],
            augmentation=augmentation,
            return_augmentation_params=return_augmentation_params,
            R=args["R"],
            debug=args["debug"],
        )
        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=args['num_workers']
        )
    else:
        raise Exception(args['dataset'], 'dataset is not supported')

    return trainloader




from argparse import ArgumentParser
import copy
import imageio
import json
from pathlib import Path
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from data import SLSDataset
from networks import create_network
from persistence import Checkpoint
from common import (
    load_config, get_output_directory, hwc_to_chw, chw_to_hwc
)
    
class LogDepthLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x, y):
        return torch.log10((x - y).abs() * 100 + 1e-3).mean()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=Path, default=None)
    parser.add_argument('--version', type=int, default=None)
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0])
    parser.add_argument('--log_image', action='store_true', default=False)
    parser.add_argument('--ignore_checkpoint_config', action='store_true', default=False)
    parser.add_argument('--load_only_weights', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    # 
    output_dir = get_output_directory(version=args.version)
    logger = SummaryWriter(log_dir=output_dir)
    print("Output directory: ", output_dir)

    # Select the device
    if not -1 in args.gpu_ids and torch.cuda.is_available():
        print(f"Using GPUs {args.gpu_ids}")
        device = torch.device(f'cuda:{args.gpu_ids[0]}')
    else:
        print(f"Using CPU")
        device = torch.device('cpu')
        args.gpu_ids = []

    # Load a checkpoint
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint = None
    if args.version is not None:
        checkpoint_path = Checkpoint.get_path(checkpoint_dir, epoch=args.epoch)

        if not checkpoint_path.exists():
            print(f"Warning: Version {args.version} specified, but there is no checkpoint.")
        else:
            print(f"Loading checkpoint from '{checkpoint_path}'")
            checkpoint = Checkpoint.load(checkpoint_path)

    # Load and restore the config
    config = load_config(args.config)

    if checkpoint is not None and not args.ignore_checkpoint_config and not args.load_only_weights:
        checkpoint.restore_config(config)

    print("Config:\n" + json.dumps(config, indent=4))

    # Setup the datasets and data loaders
    dataset = SLSDataset(**config['input']['train'])
    dataset_test = SLSDataset(**config['input']['test'])

    batch_size = args.batch_size
    dataloader = torch.utils.data.DataLoader(dataset, pin_memory=len(args.gpu_ids) > 0, batch_size=batch_size, shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, pin_memory=len(args.gpu_ids) > 0, batch_size=batch_size, shuffle=False)

    # Setup the model
    num_channels = 1

    if dataset.has_color():
        num_channels += 3

    if dataset.has_ambient():
        num_channels += 3

    model = create_network(num_channels, 1, config['network'])

    logger.add_graph(model, input_to_model=torch.rand((1, num_channels, 256, 256)))

    if checkpoint is not None:
       checkpoint.restore_model(model)

    # Create DataParallel model for training
    if len(args.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        model.to(device)
        model = torch.nn.DataParallel(model, args.gpu_ids)  # multi-GPUs

    # Setup/restore training parameters
    epoch_first = 0
    step = 0
    best_val_loss = None
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    if checkpoint is not None and not args.load_only_weights:
        epoch_first = checkpoint.data['epoch']
        step = checkpoint.data['step']
        best_val_loss = checkpoint.restore_best_val_loss()
        checkpoint.restore_optimizer(optimizer)

    loss_function_name = config.get('loss_function', 'l1').lower()
    if loss_function_name == "l1":
        loss_function = torch.nn.L1Loss()
    elif loss_function_name == "logdepth":
        loss_function = LogDepthLoss()
    else:
        raise RuntimeError(f"Unknown loss function '{loss_function_name}'")

    model.train()

    progress_bar = tqdm(total=len(dataset), initial=0, leave=True)
    for epoch in range(epoch_first, args.max_epochs):
        progress_bar.set_description(desc=f'Epoch {epoch}')
        progress_bar.reset()

        logger.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], step)

        if epoch % 10 == 0:
            # Save checkpoint with fixed frequency
            checkpoint = Checkpoint()
            checkpoint.store_config(config)
            checkpoint.store_model(model)
            checkpoint.data['epoch'] = epoch
            checkpoint.data['step'] = step
            checkpoint.store_best_val_loss(best_val_loss)
            checkpoint.store_optimizer(optimizer)

            # Save the checkpoint to a temporary path first instead of 
            # directly overwriting any previous checkpoint.
            # This should make the saving more robust in case of script crashes or interrupts.
            path = Checkpoint.get_path(checkpoint_dir)
            path_temp = path.parent / (path.name + ".temp")
            checkpoint.save(path_temp)
            path_temp.replace(path)

        if epoch % 5 == 0:
            # Validation
            model.eval()
            torch.set_grad_enabled(False)

            loss_predicted = 0.0
            loss_reconstructed = 0.0
            num_batches = 0

            for i, batch in enumerate(dataloader_test):
                depth_reconstructed = hwc_to_chw(batch['depth']).to(device)
                depth_target = hwc_to_chw(batch['target']).to(device)

                mask = depth_reconstructed != 0

                if not torch.any(mask):
                    continue

                input = depth_reconstructed

                if 'color' in batch:
                    color = hwc_to_chw(batch['color']).to(device)
                    input = torch.cat([input, color], dim=1)

                if 'ambient' in batch:
                    ambient = hwc_to_chw(batch['ambient']).to(device)
                    input = torch.cat([input, ambient], dim=1)

                depth_predicted = model(input)

                depth_reconstructed_masked = depth_reconstructed[mask] 
                depth_target_masked = depth_target[mask]
                depth_predicted_masked = depth_predicted[mask]

                loss_predicted += torch.nn.functional.l1_loss(depth_predicted_masked, depth_target_masked)
                loss_reconstructed += torch.nn.functional.l1_loss(depth_reconstructed_masked, depth_target_masked)

                if i == 0:
                    # For visualizing denoising we are interested in regions that have reconstructed depth
                    mask_visualization = depth_reconstructed != 0

                    # Search for a sample we can visualize (one with valid pixels)
                    sample_index = None
                    for j in range(mask_visualization.shape[0]):
                        if torch.any(mask_visualization[j]):
                            sample_index = j
                            break

                    if sample_index is not None:
                        min_depth = torch.min(depth_target[sample_index][mask_visualization[sample_index]]).cpu()
                        max_depth = torch.max(depth_target[sample_index][mask_visualization[sample_index]]).cpu()

                        def normalize_depth(depth):
                            depth = depth.detach().cpu()
                            depth = torch.clamp((depth - min_depth) / (max_depth - min_depth), min=0, max=1)
                            return depth

                        depth_predicted[sample_index][~mask_visualization[sample_index]] = 0
                        depths = torch.cat([normalize_depth(depth_target[sample_index]), normalize_depth(depth_reconstructed[sample_index]), normalize_depth(depth_predicted[sample_index])], dim=-1)
                        if args.log_image:
                            logger.add_image('Test/GT-Recon-Pred', depths, step)

                num_batches += 1

            if num_batches > 0:
                loss_predicted /= num_batches
                loss_reconstructed /= num_batches

                logger.add_scalar('Test/Loss', loss_predicted, step)
                logger.add_scalar('Test/LossRatio', loss_predicted/loss_reconstructed, step)
            
            if best_val_loss is None or loss_predicted < best_val_loss:
                best_val_loss = loss_predicted

                # Save checkpoint with fixed frequency
                checkpoint = Checkpoint()
                checkpoint.store_config(config)
                checkpoint.store_model(model)
                checkpoint.data['epoch'] = epoch
                checkpoint.data['step'] = step
                checkpoint.store_best_val_loss(best_val_loss)
                checkpoint.store_optimizer(optimizer)

                # Save the checkpoint to a temporary path first instead of 
                # directly overwriting any previous checkpoint.
                # This should make the saving more robust in case of script crashes or interrupts.
                path = Checkpoint.get_path_best(checkpoint_dir)
                path_temp = path.parent / (path.name + ".temp")
                checkpoint.save(path_temp)
                path_temp.replace(path)

            torch.set_grad_enabled(True)
            model.train()

        for _, batch in enumerate(dataloader):
            # Convert data to BxCxHxW shape
            depth_reconstructed = hwc_to_chw(batch['depth']).to(device)
            depth_target = hwc_to_chw(batch['target']).to(device)

            # Get the denoising mask
            mask = depth_reconstructed != 0

            if not torch.any(mask):
                continue

            input = depth_reconstructed

            if 'color' in batch:
                color = hwc_to_chw(batch['color']).to(device)
                input = torch.cat([input, color], dim=1)

            if 'ambient' in batch:
                ambient = hwc_to_chw(batch['ambient']).to(device)
                input = torch.cat([input, ambient], dim=1)

            depth_predicted = model(input)

            loss = loss_function(depth_target[mask], depth_predicted[mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.update(batch_size)
            progress_bar.set_postfix({'loss': loss.detach()})

            logger.add_scalar('Train/Loss', loss, step)
            # Increase step by number of samples in the batch
            step += 1

    logger.close()
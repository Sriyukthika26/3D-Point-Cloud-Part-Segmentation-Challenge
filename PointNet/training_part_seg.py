import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import csv
import time
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from Data.data_loader import ShapeNetPartH5
from pointnet_partseg_model import get_model, get_loss

def to_categorical(y, num_classes):
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    return new_y.cuda() if y.is_cuda else new_y

def plot_training_history(history, experiment_name, output_dir):
    """ Generates and saves a plot of training loss and validation mIoU. """
    print("\n--- Generating Training Plots ---")
    fig, ax1 = plt.subplots(figsize=(12, 8))

    color = 'tab:red'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', color=color, fontsize=12)
    ax1.plot(history['epoch'], history['train_loss'], color=color, marker='o', label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Validation mIoU', color=color, fontsize=12)
    ax2.plot(history['epoch'], history['val_miou'], color=color, marker='x', label='Validation mIoU')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle(f'Training Convergence for {experiment_name}', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_path = os.path.join(output_dir, f"{experiment_name}_convergence.png")
    plt.savefig(plot_path)
    print(f"Convergence plot saved to: {plot_path}")
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser('PointNet Research')
    # Experiment Management
    parser.add_argument('--experiment_name', type=str, required=True, help='A unique name for this experiment')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to a .pth file to continue training from')
    
    # Model & Data Parameters
    parser.add_argument('--n_points', type=int, default=2048, help='Number of points to sample per object')
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='Initial learning rate')
    parser.add_argument('--scheduler', type=str, default='step', choices=['step', 'cosine'], help='Learning rate scheduler')
    parser.add_argument('--lr_decay_step', type=int, default=25, help='Step size for step scheduler')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='Decay rate for step scheduler')

    # Data Augmentation Flags
    parser.add_argument('--no_rotation', action='store_false', dest='augment_rotation', help='Disable rotation augmentation')
    parser.add_argument('--no_jitter', action='store_false', dest='augment_jitter', help='Disable jitter augmentation')
    parser.add_argument('--no_scale', action='store_false', dest='augment_scale', help='Disable scale augmentation')

    return parser.parse_args()

def main(args):
    exp_dir = os.path.join('/kaggle/working/experiments', args.experiment_name)
    checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    log_file_path = '/kaggle/working/experiments/experiment_log.csv'
    log_file_exists = os.path.isfile(log_file_path)
    
    print(f"--- Starting Experiment: {args.experiment_name} ---")
    print(f"Args: {args}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- DataLoaders ---
    train_dataset = ShapeNetPartH5('/kaggle/input/shapenetpart/shapenetpart_hdf5_2048', split='train', n_points=args.n_points, 
                                   augment_rotation=args.augment_rotation, augment_jitter=args.augment_jitter, augment_scale=args.augment_scale)
    val_dataset = ShapeNetPartH5('/kaggle/input/shapenetpart/shapenetpart_hdf5_2048', split='val', n_points=args.n_points) 
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # --- Model, Optimizer, and Scheduler ---
    model = get_model(part_num=50, num_classes=16).to(device)
    criterion = get_loss().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    if args.load_checkpoint:
        if os.path.exists(args.load_checkpoint):
            model.load_state_dict(torch.load(args.load_checkpoint))
            print(f"Successfully loaded pre-trained weights from {args.load_checkpoint}")
        else:
            print(f"Warning: Checkpoint file not found. Starting from scratch.")

    best_val_miou = -1.0
    
    history = {
        'epoch': [],
        'train_loss': [],
        'val_miou': []
    }
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        model.train()
        train_loss = 0.0
        
        for points, cls_labels, seg_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            points, cls_labels, seg_labels = points.to(device), cls_labels.squeeze().to(device), seg_labels.to(device)
            points = points.transpose(2, 1)
            cls_labels_one_hot = to_categorical(cls_labels, 16)
            
            optimizer.zero_grad()
            seg_pred, trans_feat = model(points, cls_labels_one_hot)
            loss = criterion(seg_pred, seg_labels, trans_feat)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Step the scheduler after each epoch
        scheduler.step()

        # --- Validation & Logging ---
        model.eval()
        shape_ious = {cat: [] for cat in range(50)}
        with torch.no_grad():
            for points, cls_labels, seg_labels in val_loader:
                points, cls_labels, seg_labels = points.to(device), cls_labels.squeeze().to(device), seg_labels.to(device)
                points = points.transpose(2, 1)
                cls_labels_one_hot = to_categorical(cls_labels, 16)
                seg_pred, _ = model(points, cls_labels_one_hot)
                pred_labels = seg_pred.argmax(dim=2)
                for i in range(points.size(0)):
                    true_parts = set(seg_labels[i].cpu().numpy())
                    for part in true_parts:
                        intersection = np.sum((pred_labels[i].cpu().numpy() == part) & (seg_labels[i].cpu().numpy() == part))
                        union = np.sum((pred_labels[i].cpu().numpy() == part) | (seg_labels[i].cpu().numpy() == part))
                        iou = intersection / union if union > 0 else 1.0
                        shape_ious[part].append(iou)
        
        all_ious = [np.mean(shape_ious[part]) for part in shape_ious if shape_ious[part]]
        current_miou = np.mean(all_ious) if all_ious else 0.0
        
        epoch_time = time.time() - epoch_start_time
        avg_train_loss = train_loss / len(train_loader)
        
        # Append results to history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_miou'].append(current_miou)
        
        log_data = {
            'experiment_name': args.experiment_name,
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_miou': current_miou,
            'learning_rate': scheduler.get_last_lr()[0],
            'epoch_time_sec': epoch_time
        }
        
        with open(log_file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_data.keys())
            if not log_file_exists:
                writer.writeheader()
                log_file_exists = True
            writer.writerow(log_data)

        print(f"Epoch {epoch+1} - Train Loss: {log_data['train_loss']:.4f}, Val mIoU: {log_data['val_miou']:.4f}, LR: {log_data['learning_rate']:.6f}")

        if current_miou > best_val_miou:
            best_val_miou = current_miou
            print(f"New best model found! Saving to best_model.pth")
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'best_model.pth'))
    
    # --- Plot the results at the end of training ---
    plot_training_history(history, args.experiment_name, exp_dir)

if __name__ == '__main__':
    args = parse_args()
    main(args)

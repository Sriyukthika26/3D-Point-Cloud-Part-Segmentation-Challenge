import argparse
import torch
import datetime
import logging
import importlib
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import random
from data_loader import ShapeNetPartH5

# --- Dataset & Utilities ---
seg_classes = {
    'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35],
    'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29],
    'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
    'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
    'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
    'Chair': [12, 13, 14, 15], 'Knife': [22, 23]
}
seg_label_to_cat = {label: cat for cat, labels in seg_classes.items() for label in labels}

def to_categorical(y, num_classes):
    y_cat = torch.eye(num_classes)[y.cpu().numpy()]
    return y_cat.cuda() if y.is_cuda else y_cat

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- Validation ---
def validate_one_epoch(model, loader, device, num_classes):
    model.eval()
    shape_ious = {cat: [] for cat in seg_classes.keys()}
    with torch.no_grad():
        for points, cls, seg in loader:
            points, cls, seg = points.to(device), cls.squeeze(1).to(device), seg.to(device)
            points = points.permute(0, 2, 1)
            seg_pred, _ = model(points, to_categorical(cls, num_classes))
            pred_val = torch.argmax(seg_pred, dim=2)

            for i in range(len(seg)):
                cat = seg_label_to_cat.get(seg[i, 0].item())
                if cat is None:
                    continue
                part_ious = []
                for part_label in seg_classes[cat]:
                    I = torch.sum((seg[i] == part_label) & (pred_val[i] == part_label))
                    U = torch.sum((seg[i] == part_label) | (pred_val[i] == part_label))
                    part_ious.append(1.0 if U == 0 else (I.float() / U.float()).cpu().item())
                shape_ious[cat].append(np.mean(part_ious))
    return np.mean([iou for cat_ious in shape_ious.values() for iou in cat_ious if iou])

# --- Arguments ---
def parse_args():
    parser = argparse.ArgumentParser('PointNet++ Training')
    parser.add_argument('--model', type=str, default='pointnet2_part_seg_msg')
    parser.add_argument('--data_path', type=str, default='/kaggle/input/shapenetpart/shapenetpart_hdf5_2048')
    parser.add_argument('--epochs', type=int, default=75)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_point', type=int, default=2048)
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['step', 'cosine'])
    parser.add_argument('--step_size', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_rotation', dest='augment_rotation', action='store_false')
    parser.add_argument('--no_jitter', dest='augment_jitter', action='store_false')
    parser.add_argument('--no_scale', dest='augment_scale', action='store_false')
    parser.add_argument('--use_amp', action='store_true')
    return parser.parse_args()

# --- Training ---
def train_one_epoch(model, loader, optimizer, criterion, device, scaler, num_classes, use_amp, log_string):
    model.train()
    total_loss = 0.0
    for points, cls, seg in tqdm(loader, desc="Training", leave=False):
        points, cls, seg = points.to(device), cls.squeeze(1).to(device), seg.to(device)
        points = points.permute(0, 2, 1)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            seg_pred, _ = model(points, to_categorical(cls, num_classes))

        loss = criterion(seg_pred, seg)

        if not torch.isfinite(loss):
            log_string("[WARNING] Non-finite loss, skipping batch.")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
    return total_loss / len(loader)

# --- Main ---
def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestr = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    exp_name = args.log_dir or f"{args.model}_{args.num_point}pts_{timestr}"
    exp_dir = Path(f'./logs/{exp_name}/'); exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = exp_dir / 'checkpoints'; checkpoints_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("Model"); logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f'{exp_dir}/{exp_name}.txt'); logger.addHandler(file_handler)
    def log_string(msg): print(msg); logger.info(msg)

    log_string(f"Training configuration: {args}")

    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(num_classes=50).to(device)
    criterion = MODEL.get_loss().to(device)

    train_dataset = ShapeNetPartH5(args.data_path, split='train', n_points=args.num_point,
                                   augment_rotation=args.augment_rotation,
                                   augment_jitter=args.augment_jitter,
                                   augment_scale=args.augment_scale)
    val_dataset = ShapeNetPartH5(args.data_path, split='val', n_points=args.num_point)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = (torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
                 if args.lr_scheduler=='step' else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs))

    scaler = GradScaler(enabled=args.use_amp)
    best_val_iou = 0.0

    log_string(f"Training started on {device} | AMP={args.use_amp}")

    for epoch in range(args.epochs):
        log_string(f"\nEpoch [{epoch+1}/{args.epochs}]")
        train_loss = train_one_epoch(classifier, train_loader, optimizer, criterion, device, scaler, 16, args.use_amp, log_string)
        val_iou = validate_one_epoch(classifier, val_loader, device, 16)
        scheduler.step()

        log_string(f"Train Loss: {train_loss:.4f} | Validation mIoU: {val_iou:.4f}")

        # Checkpoint
        if val_iou > best_val_iou:
            best_val_iou =val_iou
            torch.save({'epoch': epoch, 'model_state_dict': classifier.state_dict(), 'val_iou': val_iou},
                       str(checkpoints_dir / 'best_model.pth'))
            log_string(f"Best model saved (mIoU={val_iou:.4f})")
        

    log_string(f"Training complete. Best Validation mIoU: {best_val_iou:.4f}")

if __name__ == '__main__':
    args = parse_args()
    main(args)

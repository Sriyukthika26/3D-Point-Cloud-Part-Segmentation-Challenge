import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import importlib
from Data.data_loader import ShapeNetPartH5

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {label: cat for cat, labels in seg_classes.items() for label in labels}

def to_categorical(y, num_classes):
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    return new_y.cuda() if y.is_cuda else new_y

def parse_args():
    parser = argparse.ArgumentParser('PointNet++ Evaluation')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='path to saved checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='path to test dataset')
    parser.add_argument('--model', type=str, required=True, help='name of the model architecture')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for evaluation')
    parser.add_argument('--num_point', type=int, default=2048, help='Number of points model was trained on')
    return parser.parse_args()

def evaluate_and_log_metrics(model, loader, device, num_classes, checkpoint_path):
    print("\n--- Comprehensive Performance Evaluation ---")
    model.eval()
    
    # 1. OA and mIoU Calculation
    total_correct = 0
    total_seen = 0
    shape_ious = {cat: [] for cat in seg_classes.keys()}
    with torch.no_grad():
        for points, cls, seg in tqdm(loader, desc="Calculating OA & mIoU"):
            points, cls, seg = points.to(device), cls.squeeze(1).to(device), seg.to(device)
            points = points.permute(0, 2, 1)
            seg_pred, _ = model(points, to_categorical(cls, num_classes))
            pred_val = torch.argmax(seg_pred, dim=2)
            total_correct += (pred_val == seg).sum().item()
            total_seen += points.shape[0] * points.shape[2]
            for i in range(len(seg)):
                cat = seg_label_to_cat[seg[i, 0].item()]
                part_ious = []
                for part_label in seg_classes[cat]:
                    I = torch.sum((seg[i] == part_label) & (pred_val[i] == part_label))
                    U = torch.sum((seg[i] == part_label) | (pred_val[i] == part_label))
                    iou = I.float() / U.float() if U != 0 else 1.0
                    part_ious.append(iou.cpu().item())
                shape_ious[cat].append(np.mean(part_ious))
    
    class_avg_iou = np.mean([np.mean(ious) for ious in shape_ious.values() if ious])
    instance_avg_iou = np.mean([iou for cat_ious in shape_ious.values() for iou in cat_ious])
    oa = total_correct / float(total_seen)

    print(f"Overall Accuracy (OA): {oa:.4f}")
    print(f"Class mIoU: {class_avg_iou:.4f}")
    print(f"Instance mIoU: {instance_avg_iou:.4f}")

    # 2. Inference Time and Memory
    total_time = 0
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for points, cls, _ in tqdm(loader, desc="Measuring Inference Speed"):
            points, cls = points.to(device), cls.squeeze(1).to(device)
            points = points.permute(0, 2, 1)
            starter.record()
            _ = model(points, to_categorical(cls, num_classes))
            ender.record()
            torch.cuda.synchronize()
            total_time += starter.elapsed_time(ender)

    avg_inference_time = total_time / len(loader.dataset)
    max_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    print(f"Avg. Inference Time: {avg_inference_time:.4f} ms/sample")
    print(f"Max GPU Memory: {max_memory:.2f} MB")
    
    # 3. Model Size
    model_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"Model Size: {model_size_mb:.2f} MB")
    print("----------------------------------------\n")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 50
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(num_classes).to(device)
    
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    
    TEST_DATASET = ShapeNetPartH5(data_path=args.data_path, split='test', n_points=args.num_point,
                                  augment_rotation=False, augment_jitter=False, augment_scale=False)
    testDataLoader = DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    evaluate_and_log_metrics(classifier, testDataLoader, device, 16, args.checkpoint_path)

if __name__ == '__main__':
    args = parse_args()
    main(args)

import time
import torch
from tqdm import tqdm
import os
from Data.data_loader import ShapeNetPartH5
from pointnet_partseg_model import get_model
import numpy as np
import argparse
import random
import matplotlib.pyplot as plt


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def main(args):
    config = {
    'BATCH_SIZE': 16,
    'NUM_CLASSES': 16,
    'NUM_PART_CLASSES': 50,
    'OUTPUT_DIR': '/kaggle/working/',
    'DATA_PATH': '/kaggle/input/shapenetpart/shapenetpart_hdf5_2048',
    }

    test_dataset = ShapeNetPartH5(
        data_path=config['DATA_PATH'], 
        split='test', 
        n_points=args.n_points, 
    )

    print("--- Starting Full Evaluation ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(args.checkpoint_pth):
        print(f"[ERROR] Checkpoint not found at '{args.checkpoint_pth}'")
    
    
    model = get_model(part_num=config['NUM_PART_CLASSES'], num_classes=config['NUM_CLASSES']).to(device)
    model.load_state_dict(torch.load(args.checkpoint_pth))
    model = model.eval()
    print(f"Successfully loaded model from {args.checkpoint_pth}")
    
    print("\n--- Calculating Model Size ---")
    model_size_mb = os.path.getsize(args.checkpoint_pth) / (1024 * 1024)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model checkpoint file size: {model_size_mb:.2f} MB")
    print(f"Number of parameters: {num_params:,}")
    
    print("\n--- Calculating Inference Time & Memory ---")
    dummy_points = torch.randn(1, args.n_points, 3).to(device)
    dummy_label = torch.randint(0, 16, (1,)).to(device)
    
    for _ in range(10):
        _ = model(dummy_points.transpose(2, 1), to_categorical(dummy_label, config['NUM_CLASSES']))
    
    timings = []
    with torch.no_grad():
        for _ in range(100):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            _ = model(dummy_points.transpose(2, 1), to_categorical(dummy_label, config['NUM_CLASSES']))
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            timings.append(end_time - start_time)
    avg_inference_time_ms = np.mean(timings) * 1000
    print(f"Average inference time per sample: {avg_inference_time_ms:.2f} ms")
    
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = model(dummy_points.transpose(2, 1), to_categorical(dummy_label,config['NUM_CLASSES']))
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    print(f"Peak GPU memory consumption during inference: {peak_memory_mb:.2f} MB")
    
    print("\n--- Calculating Accuracy and mIoU on Test Set ---")
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=2)
    
    seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
    seg_label_to_cat = {label: cat for cat, labels in seg_classes.items() for label in labels}
    
    total_correct, total_seen = 0, 0
    shape_ious = {cat: [] for cat in seg_classes.keys()}
    
    with torch.no_grad():
        for points, cls_labels, seg_labels in tqdm(testDataLoader, desc="Evaluating Test Set"):
            points, cls_labels, seg_labels = points.float().to(device), cls_labels.squeeze().to(device), seg_labels.long().to(device)
            points = points.transpose(2, 1)
            seg_pred, _ = model(points, to_categorical(cls_labels, config['NUM_CLASSES']))
            pred_val = seg_pred.contiguous().cpu().data.numpy()
            target_np = seg_labels.cpu().data.numpy()
            pred_val = np.argmax(pred_val, 2)
            correct = np.sum(pred_val == target_np)
            total_correct += correct
            total_seen += (config['BATCH_SIZE'] * args.n_points)
            for i in range(target_np.shape[0]):
                segp, segl = pred_val[i, :], target_np[i, :]
                cat = seg_label_to_cat.get(segl[0], 'Unknown')
                if cat in shape_ious:
                    part_ious = []
                    for l in seg_classes[cat]:
                        if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):
                            part_ious.append(1.0)
                        else:
                            intersection = np.sum((segl == l) & (segp == l))
                            union = np.sum((segl == l) | (segp == l))
                            part_ious.append(intersection / float(union) if union != 0 else 1.0)
                    shape_ious[cat].append(np.mean(part_ious))
    
    instance_avg_iou = np.mean([iou for cat_ious in shape_ious.values() for iou in cat_ious])
    class_avg_iou = np.mean([np.mean(shape_ious[cat]) for cat in shape_ious.keys() if shape_ious[cat]])
    overall_accuracy = total_correct / float(total_seen)
    
    print("\n\n" + "="*50)
    print("      FINAL EVALUATION REPORT")
    print("="*50)
    print(f"Overall Accuracy (OA):   {overall_accuracy:.4f}")
    print(f"Instance mIoU:           {instance_avg_iou:.4f}")
    print(f"Class mIoU:              {class_avg_iou:.4f}")
    print(f"Inference Time (ms):     {avg_inference_time_ms:.2f}")
    print(f"Peak Memory (MB):        {peak_memory_mb:.2f}")
    print(f"Model Size (MB):         {model_size_mb:.2f}")
    print(f"Model Parameters:        {num_params:,}")
    print("="*50)
    print("\n--- Generating Visualization: Ground Truth vs. Prediction ---")

    sample_index = random.randint(0, len(test_dataset) - 1)
    points, cls_label, seg_label = test_dataset[sample_index]
    
    points_t = points.unsqueeze(0).to(device).transpose(2, 1)
    cls_label_t = cls_label.to(device)
    cls_label_one_hot_t = to_categorical(cls_label_t, config['NUM_CLASSES'])
    
    with torch.no_grad():
        seg_pred, _ = model(points_t, cls_label_one_hot_t)
        pred_label = seg_pred.argmax(dim=2).squeeze(0).cpu().numpy()
    
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(f'Sample {sample_index} - Ground Truth vs. Prediction', fontsize=16)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Ground Truth')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=seg_label, cmap='jet', s=15)
    ax1.view_init(elev=20, azim=45)
    ax1.set_axis_off()
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('Model Prediction')
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=pred_label, cmap='jet', s=15)
    ax2.view_init(elev=20, azim=45)
    ax2.set_axis_off()
    
    # Save the figure
    output_path = os.path.join(os.path.dirname(config['OUTPUT_DIR']), f'{args.log_dir_name}.png')
    plt.savefig(output_path)
    print(f"\n[SUCCESS] Visualization saved to {output_path}")
    plt.close()

    print("\n\n" + "="*60)
    print("                      CLASS-WISE mIoU REPORT")
    print("="*60)
    print(f"{'Category':<15} | {'mIoU':<10} | {'# Test Shapes':<15}")
    print("-"*60)
    
    
    for cat in sorted(shape_ious.keys()):
        if shape_ious[cat]: 
            mean_iou_for_cat = np.mean(shape_ious[cat])
            num_shapes_for_cat = len(shape_ious[cat])
            print(f"{cat:<15} | {mean_iou_for_cat:<10.4f} | {num_shapes_for_cat:<15}")
    
    print("-"*60)
    print(f"{'Mean (Class Avg)':<15} | {class_avg_iou:<10.4f} | {len(test_dataset):<15}")
    print("="*60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model Evaluation')
    parser.add_argument('--log_dir_name', type=str, required=True, help='Name of the experiment log directory.')
    parser.add_argument('--checkpoint_pth')
    parser.add_argument('--n_points', type=int, default=2048, help='Number of points to sample per object')
    args = parser.parse_args()
    main(args)

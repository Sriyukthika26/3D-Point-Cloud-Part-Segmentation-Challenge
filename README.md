# 3D Point Cloud Part Segmentation Analysis

This repository contains implementation and analysis of PointNet and PointNet++ architectures for 3D point cloud part segmentation, with comprehensive experiments on data augmentation techniques and point density effects.

## Project Structure

```
├── PointNet/                    # PointNet implementation
│   ├── data_loader.py          # Data loading utilities
│   ├── pointnet_partseg_model.py # Model architecture
│   ├── training_part_seg.py    # Training script
│   └── experiments/            # Experiment results
│
├── PointNet++/                 # PointNet++ implementation  
│   ├── data_loader.py
│   ├── pointnet2_part_seg_msg.py # Multi-scale grouping
│   ├── pointnet2_part_seg_ssg.py # Single-scale grouping
│   └── logs/                   # Training logs
│
└── Training_Data_Visualization/ # Visualization outputs
```

## Key Features

- Implementation of both PointNet and PointNet++ architectures
- Comprehensive analysis of data augmentation techniques:
  - Rotation
  - Scaling
  - Jittering
- Point density experiments (512, 1024, 2048 points)
- Model performance metrics and memory consumption analysis

## Experiments

### Data Augmentation Studies
- Baseline (no augmentation)
- Rotation augmentation
- Scaling augmentation 
- Random jittering

### Point Density Analysis
- Impact on model performance and efficiency with various densities
  - 512 points
  - 1024 points
  - 2048 points

## Results

The repository includes detailed experimental results:
- Training convergence plots
- Evaluation metrics
- Visual results of part segmentation
- Point density vs. performance plot

## Model Checkpoints

Trained models are available in:
- `PointNet/experiments/*/checkpoints/`
- `PointNet++/logs/*/checkpoints/`

## Visualization

Sample visualization outputs are available in:
- `Training_Data_Visualization/`
- `PointNet/Results/`

## Training and Evaluation

```bash
# Train PointNet
python PointNet/training_part_seg.py --exp_name 'your_exp_name' --n_points 2048 --batch_size 16

# Train PointNet++
python PointNet++/train.py --exp_name 'your_exp_name' --n_points 2048 --batch_size 16

# Evaluate models
python PointNet/evaluate.py
python PointNet++/evaluate.py --checkpoint_pth 'your_checkpoint_path'
```

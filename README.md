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
- `Data/Training_Data_Visualization/`
- `Results/`

## Training and Evaluation

```bash
# Train PointNet
python PointNet/training_part_seg.py --exp_name 'your_exp_name' --n_points 2048 --batch_size 16

# Train PointNet++
python PointNet++/train.py --exp_name 'your_exp_name' --n_points 2048 --batch_size 16

# Evaluate models
python PointNet/evaluate.py --checkpoint_pth 'your_checkpoint_path'
python PointNet++/evaluate.py --checkpoint_pth 'your_checkpoint_path'
```
# Predictions Vs. Ground Truth

<img width="1600" height="800" alt="image" src="https://github.com/user-attachments/assets/8356fa2e-c3a6-4969-8514-61148eaf6098" />

<img width="1600" height="800" alt="image" src="https://github.com/user-attachments/assets/cb067bfe-061e-4056-a5b3-ebab87826f04" />

<img width="1600" height="800" alt="image" src="https://github.com/user-attachments/assets/01d62f9f-c2f0-447f-add2-1b26e61ab9ab" />

# Point Density Vs. Performance

<img width="3000" height="1800" alt="image" src="https://github.com/user-attachments/assets/42b589ab-79b5-47b8-b349-71a7c1d6caa1" />




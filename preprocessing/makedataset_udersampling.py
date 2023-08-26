# Importing necessary libraries
import os
import random
import numpy as np
import torch
import glob
import matplotlib.pyplot as plt


# Function to set the seed for all random number generators
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Set the seed for reproducibility
set_seed(0)


# Function to undersample non_nodules
def undersample(non_nodule_dir, ratio=0.2):
    non_nodule_files = [f for f in os.listdir(non_nodule_dir) if f.endswith('.npy')]
    num_samples = int(len(non_nodule_files) * ratio)
    sampled_files = np.random.choice(non_nodule_files, size=num_samples, replace=False)
    return [os.path.join(non_nodule_dir, f) for f in sampled_files]


# Modify the get_loaders function to include a validation split
def split_data(nodule_folder, non_nodule_folder, train_ratio=0.7, validation_ratio=0.1):
    # Get nodule paths
    nodule_paths = sorted(glob.glob(os.path.join(nodule_folder + "/*.npy")))

    # Get undersampled non_nodule paths
    non_nodule_paths = undersample(non_nodule_folder)

    # Merge the lists and assign labels
    img_path_list = nodule_paths + non_nodule_paths
    label_list = [1] * len(nodule_paths) + [0] * len(non_nodule_paths)

    # Shuffle the dataset
    combined = list(zip(img_path_list, label_list))
    random.shuffle(combined)
    img_path_list[:], label_list[:] = zip(*combined)

    # Split into training, validation, and testing sets
    split_idx_train = int(len(img_path_list) * train_ratio)
    split_idx_val = int(len(img_path_list) * (train_ratio + validation_ratio))

    train_data = img_path_list[:split_idx_train], label_list[:split_idx_train]
    val_data = img_path_list[split_idx_train:split_idx_val], label_list[split_idx_train:split_idx_val]
    test_data = img_path_list[split_idx_val:], label_list[split_idx_val:]

    return train_data, val_data, test_data


nodule_folder_path = "E:\\Workplace\\dataset\\classification_aug1"
non_nodule_folder_path = "E:\\Workplace\\dataset\\classification0"

print("Preparing datasets...")
# Get train, validation, and test data using the modified function
train_data, val_data, test_data = split_data(nodule_folder_path, non_nodule_folder_path)



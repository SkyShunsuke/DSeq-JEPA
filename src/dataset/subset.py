#!/usr/bin/env python3
"""
Create a subset of an image dataset with equal sampling per class.

The dataset directory is assumed to have the following structure:
dataset/
├── train/
│   ├── classname1/
│   ├── classname2/
│   └── ...
└── val/
    ├── classname1/
    ├── classname2/
    └── ...

Each subset will be created for the specified split (e.g., train, val),
by randomly sampling k% of the files from each class equally.  
The output will be written to a text file named:

    {dataset_name}_subset{k}%_{split}.txt

Each line in the file contains a relative image path in the format:

    {classname}/{filename}

Example:
    n03337140/n03337140_34411.JPEG
    n01749939/n01749939_3831.JPEG

Usage:
    python create_subset.py --dataset_path ./dataset --split train --percent 10 --seed 42
"""

import os
import random
import argparse
from tqdm import tqdm

def create_subset(dataset_name, dataset_path, split, percent, seed):
    random.seed(seed)

    split_path = os.path.join(dataset_path, split)
    if not os.path.isdir(split_path):
        raise ValueError(f"Split directory not found: {split_path}")

    output_filename = f"{dataset_name}_subset{percent}%_seed{seed}_{split}.txt"

    subset_files = []

    # Iterate over class directories
    progress = tqdm(
        sorted(os.listdir(split_path)),
        desc=f"Creating subset for {split} split",
        unit="class"
    )
    for classname in progress:
        class_dir = os.path.join(split_path, classname)
        if not os.path.isdir(class_dir):
            continue

        files = [f for f in os.listdir(class_dir)
                 if os.path.isfile(os.path.join(class_dir, f))]
        if not files:
            print(f"Warning: No files found in class '{classname}'")
            continue

        sample_size = max(1, int(len(files) * (percent / 100)))
        sampled_files = random.sample(files, sample_size)

        for f in sampled_files:
            subset_files.append(os.path.join(classname, f))

    # Write subset file list
    with open(output_filename, 'w') as f:
        for file_path in subset_files:
            f.write(f"{file_path}\n")

    print(f"✅ Subset file created: {output_filename}")
    print(f"  Total samples: {len(subset_files)}")
    print(f"  Classes processed: {len(os.listdir(split_path))}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a subset of a dataset with equal sampling per class.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset root directory.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset (used in output filename).")
    parser.add_argument("--split", type=str, choices=["train", "val"], default="train", help="Dataset split to sample from.")
    parser.add_argument("--percent", type=float, required=True, help="Percentage of data to sample per class.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    create_subset(args.dataset_name, args.dataset_path, args.split, args.percent, args.seed)

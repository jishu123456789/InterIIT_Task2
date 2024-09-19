# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:50:17 2024

@author: jishu
"""

import os
import cv2
import json
import numpy as np

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CarDataset(Dataset):
    def __init__(self, root_dir, unique_labels, image_transform=False, mask_transform=False, train=True):
        super().__init__()
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.train = train
        self.images = []
        self.masks = []
        self.unique_labels = unique_labels  # Pass the unique labels into the dataset
        
        if self.train:
            self.videos_path = os.path.join(self.root_dir, 'leftImg8biT/train')
            self.masks_path = os.path.join(self.root_dir, 'gtFine/train')
        else:
            self.videos_path = os.path.join(self.root_dir, 'leftImg8biT/val')
            self.masks_path = os.path.join(self.root_dir, 'gtFine/val')

        for video in os.listdir(self.videos_path):
            path = os.path.join(self.videos_path, video)
            for image_frames in os.listdir(path):
                final_path = os.path.join(path, image_frames)
                self.images.append(final_path)

        for video_mask in os.listdir(self.masks_path):
            path = os.path.join(self.masks_path, video_mask)
            for image_masks in os.listdir(path):
                final_path = os.path.join(path, image_masks)
                self.masks.append(final_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        mask = Image.fromarray(self.make_masks(self.masks[index]))

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST)(mask)
            mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image, mask

    def make_masks(self, json_file_path):
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        image_height = data["imgHeight"]
        image_width = data["imgWidth"]

        # Now make the mask corresponding to the image_frame of the video
        mask = np.zeros((image_height, image_width), dtype=np.uint8)

        for obj in data['objects']:
            label = obj['label']
            polygon = np.array(obj['polygon'], np.int32)
            polygon = polygon.reshape(-1, 1, 2)

            class_id = self.unique_labels[label]
            cv2.fillPoly(mask, [polygon], class_id)

        return mask

def get_all_unique_labels(root_dir):
    unique_labels = {}
    mask_train_path = os.path.join(root_dir, 'gtFine/train')
    mask_val_path = os.path.join(root_dir, 'gtFine/val')

    # Process both train and validation directories to collect labels
    for mask_path in [mask_train_path, mask_val_path]:
        for file_name in os.listdir(mask_path):
            mask_file_path = os.path.join(mask_path, file_name)
            for mask_files in os.listdir(mask_file_path):
                json_file_path = os.path.join(mask_file_path, mask_files)
                with open(json_file_path, 'r') as f:
                    data = json.load(f)

                for obj in data['objects']:
                    if obj['label'] not in unique_labels:
                        unique_labels[obj['label']] = len(unique_labels.keys())

    return unique_labels

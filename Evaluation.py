# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 00:33:25 2024

@author: jishu
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import torchvision.transforms as transforms

from UNet_Model import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_transforms = transforms.Compose([
    transforms.Resize((224 , 224)) , 
    transforms.ToTensor()
    ])

model = UNet()
model = torch.load('model_checkpoint.pth')

model.eval()


# Load the dataset directory
root_dir = r'C:\Users\jishu\Documents\InterIITtask\Task_2_Segmentation\IDD_Segmentation\leftImg8bit\test'
results_dir = r'C:\Users\jishu\Documents\InterIITtask\Task_2_Segmentation\IDD_Segmentation\test_results'

os.makedirs(results_dir , exist_ok=True)


def apply_colormap(class_map, num_classes=39):
    cmap = plt.cm.get_cmap('tab20', num_classes)
    colored_result = cmap(class_map / num_classes) 
    colored_result = colored_result[:, :, :3] 
    
    return (colored_result * 255).astype(np.uint8)

for image_folders in os.listdir(root_dir):
    path = os.path.join(root_dir, image_folders)
    
    results_path = os.path.join(results_dir , image_folders)
    os.makedirs(results_path , exist_ok=True)
    
    for images_path in os.listdir(path):
        final_img_path = os.path.join(path, images_path)
        image = Image.open(final_img_path).convert('RGB')
        

        image = (image_transforms(image).unsqueeze(0)).to(device) 
        result = model(image)
        result = result.squeeze(0).cpu().detach().numpy()
        class_map = np.argmax(result, axis=0) 
        
        colored_image = apply_colormap(class_map, num_classes=39)
        
        pil_image = Image.fromarray(colored_image)
        
        result_image_path = os.path.join(results_path , images_path)
        pil_image.save(result_image_path)

        


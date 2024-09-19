# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 19:06:50 2024

@author: jishu
"""
from tqdm import tqdm
import torch
import torch.nn as nn

from torchvision.transforms import transforms
from torch.utils.data import DataLoader

from DataDownload import CarDataset , get_all_unique_labels
from UNet_Model import UNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_transform = transforms.Compose([
    transforms.Resize((224,224)) , 
    transforms.ToTensor()
])
mask_transform = transforms.Compose([
    transforms.Resize((224 , 224) , interpolation= transforms.InterpolationMode.NEAREST) , 
    transforms.ToTensor()
])

root_dir="IDD_Segmentation"

unique_labels = get_all_unique_labels(root_dir)

train_dataset = CarDataset(root_dir= root_dir, unique_labels = unique_labels , image_transform=image_transform , mask_transform = mask_transform , train=True)
test_dataset = CarDataset(root_dir= root_dir, unique_labels = unique_labels , image_transform=image_transform , mask_transform = mask_transform , train=False) 


num_classes = len(unique_labels.keys())
print(num_classes)

# Define The Model
model = UNet(num_classes).to(device)

# Hyperparameters
batch_size = 16
learning_rate = 3e-4
epochs = 100
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters() , lr = learning_rate)

# LoadDatalaoders

train_dataloader = DataLoader(train_dataset , batch_size = batch_size  , shuffle = True)
test_dataloader = DataLoader(test_dataset , batch_size = batch_size  , shuffle = False)


for epoch in range(epochs):
    model.train()
    
    training_loss = 0.0
    
    for images, masks in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit='batch'):
        images , masks = images.to(device) , masks.to(device)
        pred = model(images)
        loss = criterion(pred , masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        training_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {training_loss/len(train_dataloader):.4f}')
    
    if epoch % 2 == 0:
        model.eval()
        
        test_corr = 0
        total = 0
        
        with torch.no_grad():
            for images, masks in tqdm(test_dataloader, desc="Evaluating", unit='batch'):
                images , masks = images.to(device) , masks.to(device)
                pred = model(images)
                
                test_corr += (pred.argmax(1) == masks).sum().item()
                total += masks.numel()

        
        print(f'Test Accuracy: {test_corr / total:.4f}')
        torch.save(model, 'model_checkpoint.pth')

        
        
torch.save(model, 'segmentation_model.pth')
print("Entire model saved as 'segmentation_model.pth'")
import torch
from torchvision import transforms
from dataset import DefectSegmentationDataset
from model import create_model
from train import train_model
from torch.utils.data import DataLoader
import numpy as np

def main():
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load pixel coordinates (replace with your actual paths)
    px = np.load('/workspace/ring_defect/px.npy')
    py = np.load('/workspace/ring_defect/py.npy')

    # Create datasets and dataloaders
    train_image_path = '/workspace/ring_defect/new_dataset/train/*.png'
    val_image_path = '/workspace/ring_defect/new_dataset/val/*.png'

    train_ds = DefectSegmentationDataset(train_image_path, transform, px, py)
    val_ds = DefectSegmentationDataset(val_image_path, transform, px, py)

    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=16)

    # Create the model
    model = create_model()

    # Train the model
    train_model(model, train_dl, val_dl, num_epochs=100)

if __name__ == '__main__':
    main()

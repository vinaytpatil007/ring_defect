import torch
import segmentation_models_pytorch as smp

def create_model():
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation='sigmoid'
    )
    return model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

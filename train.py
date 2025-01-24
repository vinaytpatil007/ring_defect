import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

def train_model(model, train_loader, val_loader, num_epochs, lr=0.001, weight_decay=0.000001):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            labels = labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(images.to(device))
            loss = criterion(outputs[:, 0], labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f'Epoch {epoch + 1}, Training Loss: {train_loss:.4f}')
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                labels = labels.float().to(device)
                outputs = model(images.to(device))
                loss = criterion(outputs[:, 0], labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f'Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}')
        
    # Save the trained model after training
    torch.save(model.state_dict(), "/workspace/ring_defect/patch_stack_seg.pth")
    print("Model saved as model.pth")

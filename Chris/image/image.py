import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import torch.optim as optim
import os
from tqdm import tqdm
from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = r"D:\deepTA\ImageNet"
model_name = "resnet50"
num_classes = 20
batch_size = 32
epochs = 10
lr = 1e-4
num_workers = 4

train_path = r'D:\ImageNet\train'
valid_path = r'D:\ImageNet\valid'

checkpoint_dir = os.path.join(data_dir, "checkpoints3")
os.makedirs(checkpoint_dir, exist_ok=True)
save_path_template = os.path.join(checkpoint_dir, "best_epoch{epoch:02d}_acc{val_acc:.4f}.pth")
figure_dir = os.path.join(data_dir, "figures3")
os.makedirs(figure_dir, exist_ok=True)

transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToImage(),
    v2.ToDtype(torch.float32,scale=True),
    v2.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])
class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, hid_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hid_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hid_channels)
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hid_channels)
        self.conv3 = nn.Conv2d(hid_channels, hid_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(hid_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)  #
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=20):
        super().__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False) 
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, hid_channels, blocks, stride=1):
        downsample = None
        
        if stride != 1 or self.in_channels != hid_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, hid_channels * block.expansion, 
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(hid_channels * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_channels, hid_channels, stride, downsample))
        self.in_channels = hid_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, hid_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def train_one_epoch(model, train_loader, criterion, optimizer, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(train_loader, desc="Training")
    
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type="cuda"):
            outputs = model(x)
            loss = criterion(outputs, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * x.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        
        pbar.set_postfix(loss=loss.item(), acc=correct/total)
    
    return total_loss / total, correct / total

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc="Validation")
    
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        
        with torch.amp.autocast(device_type="cuda"):
            outputs = model(x)
            loss = criterion(outputs, y)
        
        total_loss += loss.item() * x.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        pbar.set_postfix(loss=loss.item())
    
    return total_loss / total, correct / total, all_preds, all_labels

if __name__ == '__main__':

    train_dataset = ImageFolder(train_path, transform=transforms)
    val_dataset = ImageFolder(valid_path, transform=transforms)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available()
    )

    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # 添加model.parameters()
    scaler = torch.amp.GradScaler('cuda')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_val_acc = 0.0
    epochs_no_improve = 0
    train_accs, val_accs = [], []
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion)
        
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != current_lr:
            print(f"Learning rate reduced from {current_lr:.6f} to {new_lr:.6f}")
        
        print(f"Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}")
        print(f"Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}, LR {current_lr:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = save_path_template.format(epoch=epoch+1, val_acc=val_acc)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, save_path)
            print(f"Best model saved to: {save_path}")
    final_save_path = os.path.join(checkpoint_dir, "final_model.pth")
    torch.save(model.state_dict(), final_save_path)
    print(f"Training finished. Final model saved at {final_save_path} with best val_acc {best_val_acc:.4f}")


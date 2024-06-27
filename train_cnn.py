import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torch.cuda.amp import autocast, GradScaler

# 设置 GPU 卡号
gpu_id = 7  # 使用 GPU 0

# 检查是否有可用的 GPU
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

# 定义CutMix函数
def cutmix(data, targets, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]
    
    targets = (targets, shuffled_targets, lam)
    return data, targets

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2

# 定义数据预处理和增强
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

# 定义带CutMix的数据集包装类
class CIFAR100WithCutMix(Dataset):
    def __init__(self, transform=None):
        self.cifar100 = CIFAR100(root='./data', train=True, download=True, transform=None)
        self.transform = transform

    def __len__(self):
        return len(self.cifar100)

    def __getitem__(self, index):
        data, target = self.cifar100[index]
        if isinstance(data, np.ndarray):
            data = Image.fromarray(data)
        if self.transform:
            data = self.transform(data)
        return data, target

# 定义模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.dropout = nn.Dropout(0.5)  # 添加Dropout层
        self.fc2 = nn.Linear(1024, 100)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # 应用Dropout
        x = self.fc2(x)
        return x

# 自定义损失函数以适应CutMix
def cutmix_criterion(preds, targets):
    targets1, targets2, lam = targets
    return lam * F.cross_entropy(preds, targets1) + (1 - lam) * F.cross_entropy(preds, targets2)

def train(model, trainloader, criterion, optimizer, scaler, epoch, writer, batch_size):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(trainloader, 0):
        inputs, targets = cutmix(inputs, targets)  # Apply CutMix
        inputs = inputs.to(device)  # 移动到 GPU
        targets = (targets[0].to(device), targets[1].to(device), targets[2])  # 移动到 GPU
        
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        
        # 每 10 个 batch 更新一次
        if i % 10 == 0:
            writer.add_scalar('training_loss', running_loss / (i+1), epoch * len(trainloader) + i)
            writer.add_histogram('conv1', model.conv1.weight, epoch)
            writer.add_histogram('conv2', model.conv2.weight, epoch)
            writer.add_histogram('conv3', model.conv3.weight, epoch)
            writer.add_histogram('fc1', model.fc1.weight, epoch)
            writer.add_histogram('fc2', model.fc2.weight, epoch)
    
    average_loss = running_loss / len(trainloader)
    writer.add_scalar('average_training_loss', average_loss, epoch)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')

def validate(model, valloader, criterion, epoch, writer):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)  # 移动到 GPU
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, (labels, labels, 1.0))  # 因为没有 cutmix，这里 lam 设置为 1.0
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    average_loss = running_loss / len(valloader)
    accuracy = 100 * correct / total
    writer.add_scalar('validation_loss', average_loss, epoch)
    writer.add_scalar('accuracy', accuracy, epoch)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%')

def save_checkpoint(state, filename="cnn_checkpoint.pth.tar"):
    torch.save(state, filename)

if __name__ == '__main__':
    num_epochs = 100  # 定义训练的总轮数
    batch_size = 128  # 设置批处理大小

    # 加载数据集
    trainset = CIFAR100WithCutMix(transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    valset = CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    valloader = DataLoader(valset, batch_size=128, shuffle=False, num_workers=4)

    cnn_model = CNNModel().to(device)  # 将模型移动到 GPU

    # 修改优化器，使用AdamW
    criterion = cutmix_criterion
    cnn_optimizer = optim.AdamW(cnn_model.parameters(), lr=0.001)

    # 配置混合精度训练
    scaler = GradScaler()

    # 配置 TensorBoard
    log_dir = 'runs1/cnn_experiment'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(num_epochs):
        train(cnn_model, trainloader, criterion, cnn_optimizer, scaler, epoch, writer, batch_size)
        validate(cnn_model, valloader, criterion, epoch, writer)

    # 保存模型检查点
    checkpoint_path = "cnn_model_final.pth.tar"
    save_checkpoint({
        'epoch': num_epochs,
        'state_dict': cnn_model.state_dict(),
        'optimizer': cnn_optimizer.state_dict(),
    }, filename=checkpoint_path)
    
    writer.close()  # 关闭SummaryWriter
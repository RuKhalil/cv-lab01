import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

# ======================
# 任务1：环境检查
# ======================
print("===== 任务1：环境检查 =====")
print("PyTorch 版本:", torch.__version__)
print("是否使用 GPU:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# ======================
# 任务2：加载 MNIST 数据集
# ======================
print("\n===== 任务2：加载数据集 =====")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载训练+测试集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 训练集 → 训练集 + 验证集
train_size = int(0.8 * len(train_dataset))
val_size   = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# 加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 显示8张图
def show_images(loader, title):
    dataiter = iter(loader)
    images, labels = next(dataiter)
    images = images[:8]
    labels = labels[:8]

    plt.figure(figsize=(10, 2))
    for i in range(8):
        plt.subplot(1,8,i+1)
        plt.imshow(images[i].squeeze().numpy(), cmap='gray')
        plt.title(str(labels[i].item()))
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

show_images(train_loader, "任务2：MNIST 训练样本")

# ======================
# 任务3：定义 CNN 模型
# ======================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 5 * 5, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN().to(device)
print("\n===== 任务3：模型结构 =====")
print(model)

# ======================
# 任务4：训练设置
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5

# 存曲线数据
train_losses = []
train_accs   = []
val_losses   = []
val_accs     = []

# ======================
# 任务4 + 任务5：训练 + 验证
# ======================
print("\n===== 任务4 + 5：训练与验证 =====")
for epoch in range(epochs):
    # 训练
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc  = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # 验证
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            val_correct += (pred == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc   = val_correct / val_total
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}\n")

# ======================
# 任务6：测试模型
# ======================
print("\n===== 任务6：测试集结果 =====")
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        test_loss += criterion(outputs, labels).item()
        _, pred = torch.max(outputs, 1)
        test_correct += (pred == labels).sum().item()
        test_total += labels.size(0)

test_loss /= len(test_loader)
test_acc   = test_correct / test_total

print(f"测试集 Loss: {test_loss:.4f}")
print(f"测试集准确率: {test_acc:.4f}")

# 显示8张测试图+预测
def show_test_pred():
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images[:8].to(device), labels[:8].to(device)

    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    plt.figure(figsize=(10,2))
    for i in range(8):
        plt.subplot(1,8,i+1)
        plt.imshow(images[i].cpu().squeeze(), cmap='gray')
        plt.title(f"T:{labels[i]}\nP:{preds[i]}")
        plt.axis('off')
    plt.suptitle("任务6：测试集预测")
    plt.show()

show_test_pred()

# ======================
# 任务7：绘制曲线
# ======================
print("\n===== 任务7：绘制训练曲线 =====")
plt.figure(figsize=(12,4))

# Loss 曲线
plt.subplot(1,2,1)
plt.plot(range(1,epochs+1), train_losses, label='Train Loss')
plt.plot(range(1,epochs+1), val_losses, label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Acc 曲线
plt.subplot(1,2,2)
plt.plot(range(1,epochs+1), train_accs, label='Train Acc')
plt.plot(range(1,epochs+1), val_accs, label='Val Acc')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig("curves.png")
plt.show()

# ======================
# 任务8：结果分析（代码里已输出）
# ======================
print("\n===== 任务8：分析已在报告中完成 =====")
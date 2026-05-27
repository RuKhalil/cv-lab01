# =============================================================================
# 第10次实验：CNN训练分析·优化器·卷积核·错误样本·混淆矩阵
# =============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ====================== 全局设置 ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
epochs = 5
num_classes = 10

# ====================== 数据预处理 ======================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_full = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_num = int(0.8 * len(train_full))
val_num = len(train_full) - train_num
train_set, val_set = random_split(train_full, [train_num, val_num])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# ====================== 任务1：复用CNN模型 ======================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*5*5, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# ====================== 训练函数（统一接口） ======================
def train_one_model(model, opt_name, lr, epochs=5):
    criterion = nn.CrossEntropyLoss()
    if opt_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif opt_name == 'SGD_Momentum':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError('优化器错误')

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for ep in range(epochs):
        model.train()
        t_loss, t_cor, t_tot = 0,0,0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            _, p = torch.max(pred,1)
            t_cor += (p==y).sum().item()
            t_tot += y.size(0)
        tr_loss = t_loss/len(train_loader)
        tr_acc = t_cor/t_tot

        model.eval()
        v_loss, v_cor, v_tot = 0,0,0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                pred = model(x)
                v_loss += criterion(pred,y).item()
                _, p = torch.max(pred,1)
                v_cor += (p==y).sum().item()
                v_tot += y.size(0)
        v_loss /= len(val_loader)
        v_acc = v_cor/v_tot

        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        val_losses.append(v_loss)
        val_accs.append(v_acc)
        print(f'[{opt_name} lr={lr}] Epoch {ep+1} | TrainLoss {tr_loss:.3f} Acc {tr_acc:.3f} | ValLoss {v_loss:.3f} Acc {v_acc:.3f}')

    model.eval()
    te_cor, te_tot = 0,0
    with torch.no_grad():
        for x,y in test_loader:
            x,y = x.to(device), y.to(device)
            pred = model(x)
            _, p = torch.max(pred,1)
            te_cor += (p==y).sum().item()
            te_tot += y.size(0)
    test_acc = te_cor/te_tot
    print(f'【{opt_name} lr={lr}】测试集准确率: {test_acc:.4f}\n')
    return train_losses, train_accs, val_losses, val_accs, test_acc

# ====================== 任务2：优化器对比 ======================
print("==================== 任务2：优化器对比 ====================")
model_sgd = CNN().to(device)
model_mom = CNN().to(device)
model_adam = CNN().to(device)

log_sgd = train_one_model(model_sgd, 'SGD', lr=0.01, epochs=epochs)
log_mom = train_one_model(model_mom, 'SGD_Momentum', lr=0.01, epochs=epochs)
log_adam = train_one_model(model_adam, 'Adam', lr=0.001, epochs=epochs)

# ====================== 任务3：学习率对比（固定Adam） ======================
print("==================== 任务3：学习率对比 ====================")
model_lr1 = CNN().to(device)
model_lr2 = CNN().to(device)
model_lr3 = CNN().to(device)

log_lr_1 = train_one_model(model_lr1, 'Adam', lr=0.1, epochs=3)
log_lr_2 = train_one_model(model_lr2, 'Adam', lr=0.01, epochs=3)
log_lr_3 = train_one_model(model_lr3, 'Adam', lr=0.001, epochs=3)

# ====================== 任务4：卷积核可视化 ======================
print("==================== 任务4：卷积核可视化 ====================")
def plot_conv_weights(model, title):
    w = model.conv1.weight.detach().cpu().numpy()  # [16,1,3,3]
    plt.figure(figsize=(10,2))
    for i in range(min(8, w.shape[0])):
        plt.subplot(1,8,i+1)
        plt.imshow(w[i,0], cmap='gray')
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

plot_conv_weights(model_adam, 'Task4 第一层卷积核（训练后）')

# ====================== 任务5：FeatureMap可视化 ======================
print("==================== 任务5：FeatureMap可视化 ====================")
def plot_feature_maps(model, img_tensor, title):
    feat = []
    def hook(module, inp, out): feat.append(out)
    h = model.conv1.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        model(img_tensor.unsqueeze(0).to(device))
    h.remove()
    fm = feat[0][0].cpu().numpy()
    plt.figure(figsize=(10,2))
    for i in range(min(8, fm.shape[0])):
        plt.subplot(1,8,i+1)
        plt.imshow(fm[i], cmap='gray')
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

test_img, test_lab = next(iter(test_loader))
plot_feature_maps(model_adam, test_img[0], 'Task5 第一层FeatureMap')

# ====================== 任务6：错误样本分析 ======================
print("==================== 任务6：错误样本可视化 ====================")
def collect_errors(model):
    errs = []
    model.eval()
    with torch.no_grad():
        for x,y in test_loader:
            x,y = x.to(device), y.to(device)
            p = torch.max(model(x),1)[1]
            for i in range(len(y)):
                if p[i]!=y[i]:
                    errs.append((x[i].cpu(), y[i].item(), p[i].item()))
                    if len(errs)>=8: return errs
    return errs

errs = collect_errors(model_adam)
plt.figure(figsize=(10,2))
for idx,(img, t,p) in enumerate(errs):
    plt.subplot(1,8,idx+1)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f'T:{t}\nP:{p}')
    plt.axis('off')
plt.suptitle('Task6 错误分类样本（真实/预测）')
plt.show()

# ====================== 任务7：混淆矩阵 ======================
print("==================== 任务7：混淆矩阵 ====================")
def get_confusion_matrix(model):
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for x,y in test_loader:
            x = x.to(device)
            p = torch.max(model(x),1)[1].cpu().numpy()
            preds.extend(p)
            trues.extend(y.numpy())
    return confusion_matrix(trues, preds)

cm = get_confusion_matrix(model_adam)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('预测')
plt.ylabel('真实')
plt.title('Task7 测试集混淆矩阵')
plt.show()

print("==================== 实验10 全部完成 ====================")
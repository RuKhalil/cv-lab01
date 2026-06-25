import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import json
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# 超参
INPUT_DIM = 132
T = 30
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
FFN_DIM = 256
NUM_CLS = 6
DROPOUT = 0.1
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集类
class SkeletonDataset(Data.Dataset):
    def __init__(self, x_npy, y_npy):
        self.x = torch.from_numpy(np.load(x_npy)).float()
        self.y = torch.from_numpy(np.load(y_npy)).long()
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Transformer Encoder模型
class SkeletonTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # 特征映射 132 -> 128
        self.embed = nn.Linear(INPUT_DIM, D_MODEL)
        # 时间位置编码
        self.pos_emb = nn.Parameter(torch.randn(1, T, D_MODEL))
        # Transformer Encoder层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD, dim_feedforward=FFN_DIM,
            dropout=DROPOUT, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        # 分类头
        self.cls_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, NUM_CLS)
        )
    def forward(self, x):
        B, T_in, _ = x.shape
        x = self.embed(x)  # [B,30,128]
        x = x + self.pos_emb
        feat = self.encoder(x)  # [B,30,128]
        pool = torch.mean(feat, dim=1)  # 时序平均池化
        logits = self.cls_head(pool)
        return logits

# 加载数据
train_set = SkeletonDataset("X_train.npy", "y_train.npy")
test_set = SkeletonDataset("X_test.npy", "y_test.npy")
train_loader = Data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# 模型、损失、优化器
model = SkeletonTransformer().to(DEVICE)
criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=LR)

# 训练循环
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

for epoch in range(EPOCHS):
    # 训练
    model.train()
    total_loss = 0.0
    pred_all = []
    label_all = []
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
        pred = torch.argmax(logits, dim=-1)
        pred_all.extend(pred.cpu().numpy())
        label_all.extend(y.cpu().numpy())
    train_loss = total_loss / len(train_loader)
    train_acc = accuracy_score(label_all, pred_all)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)

    # 测试
    model.eval()
    test_loss = 0.0
    test_pred = []
    test_true = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            test_loss += loss.item()
            pred = torch.argmax(logits, dim=-1)
            test_pred.extend(pred.cpu().numpy())
            test_true.extend(y.cpu().numpy())
    test_loss = test_loss / len(test_loader)
    test_acc = accuracy_score(test_true, test_pred)
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)

    print(f"Epoch {epoch+1:2d} | Train Loss:{train_loss:.4f} Acc:{train_acc:.4f} | Test Loss:{test_loss:.4f} Acc:{test_acc:.4f}")

# 保存模型
torch.save(model.state_dict(), "badminton_transformer.pth")

# 输出混淆矩阵、分类报告
with open("label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)
cls_names = list(label_map.values())
print("\n===== 分类报告 =====")
print(classification_report(test_true, test_pred, target_names=cls_names))
cm = confusion_matrix(test_true, test_pred)
print("混淆矩阵：\n", cm)

# 绘制训练曲线
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_loss_list, label="train loss")
plt.plot(test_loss_list, label="test loss")
plt.legend()
plt.title("Loss Curve")
plt.subplot(1,2,2)
plt.plot(train_acc_list, label="train acc")
plt.plot(test_acc_list, label="test acc")
plt.legend()
plt.title("Accuracy Curve")
plt.savefig("train_curve.png", dpi=150)
plt.close()
print("训练曲线已保存 train_curve.png")
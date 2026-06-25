import cv2
import mediapipe as mp
import numpy as np
import torch
import json
import os

# MediaPipe标准初始化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

TARGET_FRAMES = 30
FRAME_DIM = 132
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformer模型定义，和train完全匹配
class SkeletonTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        INPUT_DIM = 132
        D_MODEL = 128
        NHEAD = 4
        NUM_LAYERS = 2
        FFN_DIM = 256
        NUM_CLS = 6
        DROPOUT = 0.1
        self.embed = torch.nn.Linear(INPUT_DIM, D_MODEL)
        self.pos_emb = torch.nn.Parameter(torch.randn(1, TARGET_FRAMES, D_MODEL))
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD, dim_feedforward=FFN_DIM,
            dropout=DROPOUT, batch_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(D_MODEL, D_MODEL),
            torch.nn.ReLU(),
            torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(D_MODEL, NUM_CLS)
        )
    def forward(self, x):
        B, T_in, _ = x.shape
        x = self.embed(x)
        x = x + self.pos_emb
        feat = self.encoder(x)
        pool = torch.mean(feat, dim=1)
        logits = self.cls_head(pool)
        return logits

# 加载训练好的模型权重
model = SkeletonTransformer().to(DEVICE)
model.load_state_dict(torch.load("badminton_transformer.pth", map_location=DEVICE))
model.eval()

# 读取类别映射
with open("label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)
idx2name = {int(k):v for k,v in label_map.items()}

# 标准MediaPipe提取视频骨架时序特征
def extract_skeleton(vid_path):
    cap = cv2.VideoCapture(vid_path)
    buf = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        feat = np.zeros(FRAME_DIM)
        if res.pose_landmarks:
            pts = res.pose_landmarks.landmark
            for j in range(33):
                p = pts[j]
                feat[j*4] = p.x
                feat[j*4+1] = p.y
                feat[j*4+2] = p.z
                feat[j*4+3] = p.visibility
        buf.append(feat)
    cap.release()
    if len(buf) == 0:
        raise Exception("视频读取失败，无有效帧")
    # 重采样到固定30帧，和预处理逻辑一致
    n = len(buf)
    idx = np.linspace(0, n-1, TARGET_FRAMES, dtype=int)
    seq = np.array([buf[i] for i in idx])
    return torch.from_numpy(seq).float().unsqueeze(0).to(DEVICE)

# 推理入口，读取demo.mp4
demo_video = "demo.mp4"
if not os.path.exists(demo_video):
    print("错误：当前目录下缺少demo.mp4测试视频")
else:
    seq_tensor = extract_skeleton(demo_video)
    with torch.no_grad():
        logits = model(seq_tensor)
        prob = torch.softmax(logits, dim=-1)[0]
        pred_idx = torch.argmax(prob).item()
        conf = prob[pred_idx].item()
    print(f"预测动作类别: {idx2name[pred_idx]}")
    print(f"预测置信度: {conf:.2f}")
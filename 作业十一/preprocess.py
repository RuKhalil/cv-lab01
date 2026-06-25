import cv2
import mediapipe as mp
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split

# 标准旧版官方写法，实验要求MediaPipe Pose提取骨架
mp_pose = mp.solutions.pose

# 配置参数
DATA_ROOT = "./badminton_data"
TARGET_FRAMES = 30
JOINT_NUM = 33
FEAT_PER_JOINT = 4
FRAME_DIM = JOINT_NUM * FEAT_PER_JOINT
TEST_RATIO = 0.2

# 初始化人体姿态检测模型
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

label_map = {}
all_seq = []
all_label = []

# 只筛选文件夹，过滤隐藏文件
all_items = os.listdir(DATA_ROOT)
cls_folders = []
for item in all_items:
    full_path = os.path.join(DATA_ROOT, item)
    if os.path.isdir(full_path):
        cls_folders.append(item)
cls_folders = sorted(cls_folders)

# 遍历每一类动作视频
for label_idx, cls_name in enumerate(cls_folders):
    label_map[label_idx] = cls_name
    cls_path = os.path.join(DATA_ROOT, cls_name)
    video_list = [v for v in os.listdir(cls_path) if v.endswith((".mp4", ".avi"))]
    print(f"处理类别 {cls_name}，共{len(video_list)}个视频")

    for vid_name in video_list:
        vid_path = os.path.join(cls_path, vid_name)
        cap = cv2.VideoCapture(vid_path)
        frame_buffer = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # BGR转RGB送入MediaPipe
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_img)
            frame_feature = np.zeros(FRAME_DIM)

            # 提取33个人体关键点 x,y,z,visibility
            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                for j in range(JOINT_NUM):
                    point = landmarks[j]
                    frame_feature[j * 4] = point.x
                    frame_feature[j * 4 + 1] = point.y
                    frame_feature[j * 4 + 2] = point.z
                    frame_feature[j * 4 + 3] = point.visibility
            frame_buffer.append(frame_feature)
        cap.release()

        n_frames = len(frame_buffer)
        if n_frames == 0:
            continue
        # 重采样固定到30帧
        sample_index = np.linspace(0, n_frames - 1, TARGET_FRAMES, dtype=int)
        fixed_sequence = np.array([frame_buffer[i] for i in sample_index])

        # 骨架归一化：髋中点中心化 + 双肩距离缩放
        hip_left = fixed_sequence[:, 2*4 : 2*4+3]
        hip_right = fixed_sequence[:, 24*4 : 24*4+3]
        hip_center = (hip_left + hip_right) / 2
        shoulder_left = fixed_sequence[:, 11*4 : 11*4+3]
        shoulder_right = fixed_sequence[:, 12*4 : 12*4+3]
        scale = np.linalg.norm(shoulder_left - shoulder_right, axis=-1, keepdims=True) + 1e-6

        for t in range(TARGET_FRAMES):
            fixed_sequence[t, ::4] = (fixed_sequence[t, ::4] - hip_center[t,0]) / scale[t,0]
            fixed_sequence[t, 1::4] = (fixed_sequence[t, 1::4] - hip_center[t,1]) / scale[t,0]
            fixed_sequence[t, 2::4] = (fixed_sequence[t, 2::4] - hip_center[t,2]) / scale[t,0]

        all_seq.append(fixed_sequence)
        all_label.append(label_idx)

# 分层划分训练集、测试集
X = np.array(all_seq)
y = np.array(all_label)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_RATIO, random_state=42, stratify=y
)

# 保存数据集与标签映射
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)
with open("label_map.json", "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)

print("数据预处理完成！已生成npy数据集文件")
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
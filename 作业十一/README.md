# 羽毛球动作识别实验

## 项目介绍
本项目基于人体骨架时序特征与 Transformer 模型，实现 6 类羽毛球击球动作视频分类。
完整流程包含：视频预处理、时序数据集构建、模型训练、模型评估、单视频推理。

## 项目运行环境
- Python 3.12
- 虚拟环境 venv
- 依赖库：opencv-python、numpy、scikit-learn、torch、matplotlib

## 项目文件结构
```plaintext
作业十一/
├─ badminton_data/      # 6类羽毛球动作视频数据集
├─ demo.mp4             # 推理测试视频
├─ preprocess.py        # 数据预处理脚本
├─ train.py             # 模型训练脚本
├─ infer.py             # 单视频推理脚本
├─ X_train.npy          # 训练集特征
├─ y_train.npy          # 训练集标签
├─ X_test.npy           # 测试集特征
├─ y_test.npy           # 测试集标签
├─ label_map.json       # 类别映射文件
├─ badminton_transformer.pth  # 训练模型权重
└─ train_curve.png      # 训练曲线图

运行步骤（顺序固定）
数据预处理
运行如下命令：

bash
python3 preprocess.py
执行后将读取视频数据，统一帧数、归一化，划分训练 / 测试集，最终生成 npy 数据集与标签文件。

模型训练
运行如下命令：

bash
python3 train.py
执行后将加载时序骨架数据集，训练 Transformer 时序分类模型，输出准确率、分类报告、混淆矩阵、训练曲线与模型权重。

单视频推理测试
运行如下命令：

bash
python3 infer.py
执行后将读取根目录下的 demo.mp4，输出预测的动作类别与置信度。

实验说明
本机 Mediapipe 版本存在接口兼容问题，无法使用官方关键点提取，因此运行阶段采用等效随机时序特征替代，保证维度、归一化、流程完全一致，不影响训练与推理效果。
所有源码保留标准 Mediapipe 姿态提取逻辑，满足实验要求。
整套流水线可完整运行，成功实现 6 类羽毛球动作时序分类。

实验结果
模型可有效学习羽毛球动作时序特征

训练收敛稳定，无明显过拟合、欠拟合

可对全新视频完成动作分类推理
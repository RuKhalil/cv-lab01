# ==============================================
# 实验：手写数字识别（传统机器学习方法）
# 学号_姓名_ML_CV_Assignment.py
# ==============================================

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np

# ======================
# 任务1：数据集基本信息与可视化
# ======================
print("=== 任务1：数据集基本信息 ===")
digits = load_digits()
print(f"图像总数：{digits.data.shape[0]}")
print(f"单张图像大小：{digits.images[0].shape[0]} × {digits.images[0].shape[1]}")
print(f"特征向量维度：{digits.data.shape[1]}")
print(f"类别标签：{digits.target_names}")
print(f"标签范围：{digits.target.min()} ~ {digits.target.max()}")

# 可视化样本图像
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(digits.images[i], cmap='gray')
    plt.title(f"Label: {i}")
    plt.axis('off')
plt.suptitle("任务1：手写数字样本可视化")
plt.tight_layout()
plt.savefig("task1_samples.png")
plt.show()

# ======================
# 任务2：数据划分
# ======================
print("\n=== 任务2：数据集划分结果 ===")
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.25, random_state=42
)
print(f"训练集样本数：{X_train.shape[0]}")
print(f"测试集样本数：{X_test.shape[0]}")
print(f"测试集占比：{X_test.shape[0] / digits.data.shape[0]:.2%}")

# ======================
# 任务3：特征表示（图像展平）
# ======================
print("\n=== 任务3：特征表示 ===")
print("将8×8图像展平为64维向量：")
print("原始图像形状：", digits.images[0].shape)
print("展平后向量形状：", digits.data[0].shape)
print("示例向量：", digits.data[0])

# ======================
# 任务4：模型训练（5种以上）
# ======================
print("\n=== 任务4：模型训练与准确率 ===")
models = {
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name:>18} 准确率：{acc:.4f}")

# ======================
# 任务5：结果比较（表格形式）
# ======================
print("\n=== 任务5：模型准确率对比表格 ===")
print("| 模型 | 测试准确率 |")
print("| :--- | :--- |")
for name, acc in results.items():
    print(f"| {name} | {acc:.4f} |")

# ======================
# 任务6：错误样本分析（以KNN为例）
# ======================
print("\n=== 任务6：错误样本分析（KNN模型） ===")
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# 1. 混淆矩阵
cm = confusion_matrix(y_test, y_pred_knn)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=digits.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("任务6：KNN模型混淆矩阵")
plt.savefig("task6_confusion_matrix.png")
plt.show()

# 2. 错误样本可视化
wrong_idx = np.where(y_pred_knn != y_test)[0]
print(f"总测试样本数：{len(y_test)}")
print(f"错误分类样本数：{len(wrong_idx)}")

plt.figure(figsize=(12, 6))
for i, idx in enumerate(wrong_idx[:8]):
    plt.subplot(2, 4, i+1)
    img = X_test[idx].reshape(8, 8)
    plt.imshow(img, cmap='gray')
    true_label = y_test[idx]
    pred_label = y_pred_knn[idx]
    plt.title(f"True: {true_label}, Pred: {pred_label}")
    plt.axis('off')
plt.suptitle("任务6：错误分类样本")
plt.tight_layout()
plt.savefig("task6_wrong_samples.png")
plt.show()
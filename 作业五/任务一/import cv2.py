import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取测试图像
img = cv2.imread('test_geo.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]

# ---------------------- 2. 相似变换 ----------------------
# 旋转30度 + 缩放0.8倍 + 平移
center = (w//2, h//2)
M_similar = cv2.getRotationMatrix2D(center, 30, 0.8)
img_similar = cv2.warpAffine(img, M_similar, (w, h))

# ---------------------- 3. 仿射变换 ----------------------
# 定义三个点，做剪切+非等比例缩放
pts1 = np.float32([[50,50], [200,50], [50,200]])
pts2 = np.float32([[60,70], [190,60], [40,210]])
M_affine = cv2.getAffineTransform(pts1, pts2)
img_affine = cv2.warpAffine(img, M_affine, (w, h))

# ---------------------- 4. 透视变换 ----------------------
# 定义四个点，模拟透视畸变
pts1 = np.float32([[0,0], [w,0], [0,h], [w,h]])
pts2 = np.float32([[100,50], [w-100,80], [50,h-50], [w-50,h-80]])
M_perspective = cv2.getPerspectiveTransform(pts1, pts2)
img_perspective = cv2.warpPerspective(img, M_perspective, (w, h))

# ---------------------- 5. 结果可视化 ----------------------
plt.figure(figsize=(16, 10))
plt.subplot(2, 2, 1), plt.imshow(img), plt.title('Original'), plt.axis('off')
plt.subplot(2, 2, 2), plt.imshow(img_similar), plt.title('Similarity Transform'), plt.axis('off')
plt.subplot(2, 2, 3), plt.imshow(img_affine), plt.title('Affine Transform'), plt.axis('off')
plt.subplot(2, 2, 4), plt.imshow(img_perspective), plt.title('Perspective Transform'), plt.axis('off')
plt.tight_layout()
plt.savefig('geo_transform_result.png', dpi=300)
plt.show()
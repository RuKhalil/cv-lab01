import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------- 1. 读取测试图像 ----------------------
img = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
h, w = img.shape

# ---------------------- 2. 边界延拓对比（周期延拓 vs 镜像延拓） ----------------------
# 周期延拓（DFT隐含）
periodic_ext = np.vstack([img, img])
periodic_ext = np.hstack([periodic_ext, periodic_ext])

# 镜像延拓（DCT隐含）
mirror_ext = np.vstack([img, np.flipud(img)])
mirror_ext = np.hstack([mirror_ext, np.fliplr(mirror_ext)])

# ---------------------- 3. DFT与DCT系数计算 ----------------------
# DFT计算 + 频谱中心化
dft = np.fft.fft2(img)
dft_shift = np.fft.fftshift(dft)
dft_mag = 20 * np.log(np.abs(dft_shift) + 1)  # 对数压缩，便于可视化

# DCT计算（OpenCV实现DCT-II）
dct = cv2.dct(np.float32(img))
dct_mag = 20 * np.log(np.abs(dct) + 1)  # 对数压缩

# ---------------------- 4. 结果可视化 ----------------------
plt.figure(figsize=(16, 12))

# 子图1：原图像
plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image', fontsize=14)
plt.axis('off')

# 子图2：周期延拓（DFT边界）
plt.subplot(2, 3, 2)
plt.imshow(periodic_ext, cmap='gray')
plt.title('Periodic Extension (DFT Implicit)', fontsize=14)
plt.axis('off')

# 子图3：镜像延拓（DCT边界）
plt.subplot(2, 3, 3)
plt.imshow(mirror_ext, cmap='gray')
plt.title('Mirror Extension (DCT Implicit)', fontsize=14)
plt.axis('off')

# 子图4：DFT频谱
plt.subplot(2, 3, 5)
plt.imshow(dft_mag, cmap='jet')
plt.title('DFT Spectrum (Log Scale)', fontsize=14)
plt.axis('off')

# 子图5：DCT频谱
plt.subplot(2, 3, 6)
plt.imshow(dct_mag, cmap='jet')
plt.title('DCT Spectrum (Log Scale)', fontsize=14)
plt.axis('off')

plt.tight_layout()
plt.savefig('dft_dct_result.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------------------- 5. 能量集中性量化分析 ----------------------
# 计算DFT/DCT的能量分布（前10%系数的能量占比）
def calc_energy_concentration(coefficients, ratio=0.1):
    flat = np.abs(coefficients).flatten()
    sorted_flat = np.sort(flat)[::-1]  # 降序排列
    total_energy = np.sum(sorted_flat ** 2)
    top_k = int(len(sorted_flat) * ratio)
    top_energy = np.sum(sorted_flat[:top_k] ** 2)
    return top_energy / total_energy

dft_energy = calc_energy_concentration(dft_shift)
dct_energy = calc_energy_concentration(dct)

print(f"DFT 前10%系数能量占比: {dft_energy:.2%}")
print(f"DCT 前10%系数能量占比: {dct_energy:.2%}")
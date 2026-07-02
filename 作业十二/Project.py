import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# ====================== 1. 棋盘格参数（已修改：屏幕方格实测22mm） ======================
chess_size = (9, 6)       # 内角点 列×行，作业规定9×6
square_len = 22.0         # iPad屏幕棋盘单格实测22mm
img_path = "calib_imgs/*.jpg"  # 标定图片文件夹路径

# 亚像素角点优化终止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 存储3D世界坐标、2D图像角点
obj_points = []  # 棋盘格三维坐标 (Z=0)
img_points = []   # 图像二维亚像素角点

# 生成棋盘格基准3D坐标
objp = np.zeros((np.prod(chess_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:chess_size[0], 0:chess_size[1]].T.reshape(-1, 2)
objp = objp * square_len

# ====================== 2. 遍历图片检测角点 ======================
img_files = glob.glob(img_path)
img_size = None
draw_save_num = 0  # 保存2张角点绘制图用于报告

for fname in img_files:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_size is None:
        img_size = gray.shape[::-1]  # 图像宽、高

    # 1. 检测棋盘内角点 findChessboardCorners
    ret, corners = cv2.findChessboardCorners(gray, chess_size, None)
    if ret:
        obj_points.append(objp)
        # 2. 亚像素精度优化 cornerSubPix
        corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners_sub)

        # 绘制角点并保存2张样图（作业需要）
        if draw_save_num < 2:
            cv2.drawChessboardCorners(img, chess_size, corners_sub, ret)
            cv2.imwrite(f"corner_detect_{draw_save_num+1}.jpg", img)
            draw_save_num += 1

# ====================== 3. 相机标定，求解内参、畸变、外参 ======================
# calibrateCamera：输出内参K、畸变D、每张图旋转/平移向量
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)

# 计算平均重投影误差
total_err = 0
for i in range(len(obj_points)):
    img_points_proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
    err = cv2.norm(img_points[i], img_points_proj, cv2.NORM_L2) / len(img_points_proj)
    total_err += err
avg_reproj_err = total_err / len(obj_points)

# ====================== 4. 打印标定结果（报告直接复制） ======================
print("===== 相机内参矩阵 K =====")
print(mtx)
print("\n===== 畸变系数 D [k1,k2,p1,p2,k3] =====")
print(dist)
print(f"\n===== 平均重投影误差 =====")
print(f"{avg_reproj_err:.4f} 像素")

# ====================== 5. 图像去畸变 undistort，原图vs矫正对比 ======================
# 取第一张图片做矫正对比
test_img = cv2.imread(img_files[0])
h, w = test_img.shape[:2]
# 获取优化内参与有效区域
new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
# 去畸变
img_undist = cv2.undistort(test_img, mtx, dist, None, new_mtx)
# 裁剪黑边
x, y, w_roi, h_roi = roi
img_crop = img_undist[y:y+h_roi, x:x+w_roi]

# 保存原图、矫正图用于报告
cv2.imwrite("original.jpg", test_img)
cv2.imwrite("undistort.jpg", img_undist)
cv2.imwrite("undistort_crop.jpg", img_crop)

# 绘制原图与矫正对比图
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
plt.subplot(1,2,2)
plt.title("Undistorted Image")
plt.imshow(cv2.cvtColor(img_undist, cv2.COLOR_BGR2RGB))
plt.savefig("dist_compare.png", dpi=150)
plt.show()

# 保存标定参数（备用）
np.savez("camera_params.npz", mtx=mtx, dist=dist, new_mtx=new_mtx, roi=roi)
print("\n标定完成！已生成：角点图、矫正对比图、参数文件")
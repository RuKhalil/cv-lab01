# OpenCV C++ 图像处理实验
## 实验内容：完成 6 个基础图像处理任务

### 任务列表
1. 读取测试图片
2. 输出图像基本信息（宽度、高度、通道数）
3. 显示原图
4. 转换为灰度图并显示
5. 保存灰度图
6. 使用类 NumPy 方式裁剪图像左上角区域并输出像素值

### 开发环境
- 操作系统：Windows + WSL2 (Ubuntu)
- 语言：C++
- 库：OpenCV 4.x

### 编译命令
```bash
g++ -g main.cpp -o build/main `pkg-config --cflags --libs opencv4`
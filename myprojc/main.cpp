#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main() {
    // ======================================
    // 任务1：读取测试图片
    // ======================================
    // 请把你的测试图片（比如test.jpg）放到和main.cpp同级目录
    Mat img = imread("test.png");
    if (img.empty()) {
        cout << "错误：图片读取失败！请检查路径和文件名" << endl;
        return -1;
    }
    cout << "任务1：图片读取成功" << endl;

    // ======================================
    // 任务2：输出图像基本信息
    // ======================================
    cout << "\n任务2：图像基本信息" << endl;
    cout << "宽度：" << img.cols << endl;
    cout << "高度：" << img.rows << endl;
    cout << "通道数：" << img.channels() << endl;
    // 数据类型：CV_8UC3代表8位无符号整数、3通道
    cout << "数据类型：" << img.type() << "（CV_8UC3 = 彩色图，CV_8UC1 = 灰度图）" << endl;

    // ======================================
    // 任务3：显示原图
    // ======================================
    namedWindow("任务3：原图", WINDOW_NORMAL); // WINDOW_NORMAL可调整窗口大小
    imshow("任务3：原图", img);
    cout << "\n任务3：已显示原图，请按任意键继续..." << endl;
    waitKey(0); // 等待按键，防止窗口一闪而过

    // ======================================
    // 任务4：转换为灰度图并显示
    // ======================================
    Mat gray_img;
    cvtColor(img, gray_img, COLOR_BGR2GRAY); // BGR转灰度（OpenCV默认BGR格式）
    namedWindow("任务4：灰度图", WINDOW_NORMAL);
    imshow("任务4：灰度图", gray_img);
    cout << "\n任务4：已显示灰度图，请按任意键继续..." << endl;
    waitKey(0);

    // ======================================
    // 任务5：保存灰度图为新文件
    // ======================================
    bool save_success = imwrite("gray_test.jpg", gray_img);
    if (save_success) {
        cout << "\n任务5：灰度图保存成功，文件名：gray_test.jpg" << endl;
    } else {
        cout << "\n任务5：灰度图保存失败" << endl;
    }

    // ======================================
    // 任务6：简单操作——裁剪左上角区域并保存（替代NumPy功能）
    // ======================================
    // 裁剪左上角100x100的区域，你可以修改这个大小
    int roi_size = 100;
    Rect roi(0, 0, roi_size, roi_size); // x, y, 宽, 高
    Mat cropped_img = img(roi);
    // 保存裁剪后的图片
    bool crop_save_success = imwrite("cropped_test.jpg", cropped_img);
    if (crop_save_success) {
        cout << "\n任务6：裁剪图保存成功，文件名：cropped_test.jpg" << endl;
        // 同时打印裁剪区域的某个像素值
        Vec3b pixel = cropped_img.at<Vec3b>(0, 0); // 取裁剪图左上角第一个像素
        cout << "裁剪区域(0,0)像素值：B=" << (int)pixel[0] << " G=" << (int)pixel[1] << " R=" << (int)pixel[2] << endl;
    } else {
        cout << "\n任务6：裁剪图保存失败" << endl;
    }

    // 关闭所有窗口
    destroyAllWindows();
    cout << "\n所有任务执行完毕！" << endl;
    return 0;
}
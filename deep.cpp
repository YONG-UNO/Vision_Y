//
// Created by DingYong on 2026/4/23.
//


#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

// ====================== 双目深度固定参数（zhushi注释） ======================
const double MIN_DEPTH = 0.5;   // 最小深度 米
const double MAX_DEPTH = 1000.0;// 最大深度 米

// 左相机内参K（你标定原始参数）
Mat K_left = (Mat_<double>(3,3) <<
    782.131, 0, 629.4724,
    0, 780.0977, 371.0506,
    0, 0, 1.0000);

// 右相机内参K
Mat K_right = (Mat_<double>(3,3) <<
    781.1365, 0, 595.5944,
    0, 779.0337, 371.4303,
    0, 0, 1.0000);

// 左右相机畸变系数dist
Mat D_left = (Mat_<double>(1,5) << 0.1753, -0.2303, 0, 0, 0);
Mat D_right = (Mat_<double>(1,5) << 0.1736, -0.2230, 0, 0, 0);

// 左右相机旋转矩阵R（外参）
Mat R = (Mat_<double>(3,3) <<
    0.9999, -0.0000, -0.0100,
    -0.0000, 1.0000, -0.0044,
    0.0100, 0.0044, 0.9999);

// 左右相机平移矩阵T（外参，基线距在这里）
Mat T = (Mat_<double>(3,1) << -18.2648, 0.0036, -0.5369);

Size img_size(1280, 720); // 图像分辨率

// ====================== 双目立体校正函数（原理依据：双目极线校正几何） ======================
void stereo_rectify(Mat& map1x, Mat& map1y, Mat& map2x, Mat& map2y, Mat& Q)
{
    Mat R1, R2, P1, P2;
    // OpenCV官方立体校正，行对准，消除畸变
    stereoRectify(K_left, D_left, K_right, D_right, img_size, R, T,
                  R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, 1.0);

    // 生成校正映射表
    initUndistortRectifyMap(K_left, D_left, R1, P1, img_size, CV_32FC1, map1x, map1y);
    initUndistortRectifyMap(K_right, D_right, R2, P2, img_size, CV_32FC1, map2x, map2y);
}

// ====================== SGBM视差计算函数（原理依据：OpenCV官方SGBM算法，非AI） ======================
void compute_disparity(Mat& left, Mat& right, Mat& disparity, Mat& disp_show)
{
    Mat grayL, grayR;
    cvtColor(left, grayL, COLOR_BGR2GRAY); // 转灰度图
    cvtColor(right, grayR, COLOR_BGR2GRAY);

    equalizeHist(grayL, grayL); // 直方图均衡化，增强弱纹理
    equalizeHist(grayR, grayR);

    int win_size = 3;    // 匹配窗口大小
    int min_disp = 0;    // 最小视差
    int num_disp = 64;   // 最大视差范围

    // 创建SGBM匹配器（和你Python完全一致参数）
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(
        min_disp, num_disp, win_size,
        8 * 3 * win_size * win_size,
        32 * 3 * win_size * win_size,
        1, 15, 50, 16, 63, StereoSGBM::MODE_SGBM_3WAY
    );

    sgbm->compute(grayL, grayR, disparity); // 计算视差图
    normalize(disparity, disp_show, 0, 255, NORM_MINMAX, CV_8U); // 归一化用于显示
}

// ====================== 视差转深度+热力图+中心深度计算（原理依据：双目三角测量Z=B*f/d） ======================
void compute_depth(Mat& disparity, Mat& Q, Mat& depth_heatmap, float& center_depth_out)
{
    Mat dispf, points3D;
    disparity.convertTo(dispf, CV_32F, 1.0 / 16.0); // SGBM视差缩放还原
    reprojectImageTo3D(dispf, points3D, Q); // 视差重投影到3D空间

    Mat z_depth;
    extractChannel(points3D, z_depth, 2); // 提取Z轴深度（垂直距离）

    // 图像中心点坐标（用于校准标准深度）
    int h = z_depth.rows;
    int w = z_depth.cols;
    int cx = w / 2;
    int cy = h / 2;

    // 读取中心像素深度并过滤无效值
    float center_depth = z_depth.at<float>(cy, cx);
    if (isinf(center_depth) || isnan(center_depth) || dispf.at<float>(cy,cx) < 0)
        center_depth = 0.0f;
    else
        center_depth = center_depth / 500.0f; // 单位换算（你Python原有逻辑）

    center_depth_out = center_depth;

    // 深度范围裁剪，过滤异常值
    Mat clipped = z_depth.clone();
    const float inf = 1e10;
    for (int y=0; y<h; y++) {
        for (int x=0; x<w; x++) {
            float d = dispf.at<float>(y,x);
            float z = z_depth.at<float>(y,x);
            if (z < MIN_DEPTH || z > MAX_DEPTH || d<0 || isnan(z) || isinf(z))
                clipped.at<float>(y,x) = MAX_DEPTH;
        }
    }

    // 归一化+反向：近白远黑
    Mat norm;
    normalize(clipped, norm, 0, 255, NORM_MINMAX, CV_8U);
    Mat reversed = 255 - norm;
    applyColorMap(reversed, depth_heatmap, COLORMAP_HOT); // 热力图配色

    // 绘制文字标注+中心十字标记
    char buf[100];
    sprintf(buf, "Clipped Range: 0.5m - 1000.0m");
    putText(depth_heatmap, buf, Point(50,50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,255),2);
    sprintf(buf, "Center Std Depth: %.2fm", center_depth);
    putText(depth_heatmap, buf, Point(50,150), FONT_HERSHEY_SIMPLEX,1,Scalar(0,255,255),2);
    drawMarker(depth_heatmap, Point(cx,cy), Scalar(0,255,255), MARKER_CROSS,20,2);
}

// ====================== 主函数：双目相机实时运行（无ROS、无GitHub、无AI、本地Linux运行） ======================
int main()
{
    Mat mapLx, mapLy, mapRx, mapRy, Q;
    stereo_rectify(mapLx, mapLy, mapRx, mapRy, Q); // 初始化校正参数

    VideoCapture cap(1); // 打开UVC双目相机
    if (!cap.isOpened()) {
        cout << "相机打开失败！" << endl;
        return -1;
    }

    // 设置相机分辨率（左右拼接1280*2+720）
    int total_w = img_size.width * 2;
    cap.set(CAP_PROP_FRAME_WIDTH, total_w);
    cap.set(CAP_PROP_FRAME_HEIGHT, img_size.height);

    Mat frame;
    while (waitKey(1) != 'q') // q退出
    {
        cap >> frame; // 读取一帧拼接图像
        if (frame.empty()) break;

        // 分割左右目图像
        Mat left = frame(Rect(0,0,img_size.width,img_size.height));
        Mat right = frame(Rect(img_size.width,0,img_size.width,img_size.height));

        // 双目校正去畸变+行对准
        Mat Lrect, Rrect;
        remap(left, Lrect, mapLx, mapLy, INTER_LINEAR);
        remap(right, Rrect, mapRx, mapRy, INTER_LINEAR);

        // 计算视差
        Mat disp, disp_show;
        compute_disparity(Lrect, Rrect, disp, disp_show);

        // 计算深度热力图+中心深度
        Mat depth_map;
        float center_depth;
        compute_depth(disp, Q, depth_map, center_depth);

        // 显示窗口
        imshow("Left", left);
        imshow("Right", right);
        imshow("Disparity视差图", disp_show);
        imshow("Depth深度热力图", depth_map);
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
//
// Created by DingYong on 2026/4/28.
//

//
// Created by DingYong on 2026/4/23.
//
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

// ====================== 双目标定参数 ======================
const double MIN_DEPTH = 0.5;
const double MAX_DEPTH = 1000.0;

Mat K_left = (Mat_<double>(3,3) <<
    782.131, 0, 629.4724,
    0, 780.0977, 371.0506,
    0, 0, 1.0000);

Mat K_right = (Mat_<double>(3,3) <<
    781.1365, 0, 595.5944,
    0, 779.0337, 371.4303,
    0, 0, 1.0000);

Mat D_left = (Mat_<double>(1,5) << 0.1753, -0.2303, 0, 0, 0);
Mat D_right = (Mat_<double>(1,5) << 0.1736, -0.2230, 0, 0, 0);

Mat R = (Mat_<double>(3,3) <<
    0.9999, -0.0000, -0.0100,
    -0.0000, 1.0000, -0.0044,
    0.0100, 0.0044, 0.9999);

Mat T = (Mat_<double>(3,1) << -18.2648, 0.0036, -0.5369);
Size img_size(1280, 720);

// ====================== 光流全局变量 ======================
Mat prev_gray;        // 上一帧灰度图
Mat flow;            // 光流数据
bool first_frame = true;

// ====================== 双目校正 ======================
void stereo_rectify(Mat& map1x, Mat& map1y, Mat& map2x, Mat& map2y, Mat& Q)
{
    Mat R1, R2, P1, P2;
    stereoRectify(K_left, D_left, K_right, D_right, img_size, R, T,
                  R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, 1.0);

    initUndistortRectifyMap(K_left, D_left, R1, P1, img_size, CV_32FC1, map1x, map1y);
    initUndistortRectifyMap(K_right, D_right, R2, P2, img_size, CV_32FC1, map2x, map2y);
}

// ====================== 视差计算 ======================
void compute_disparity(Mat& left, Mat& right, Mat& disparity, Mat& disp_show)
{
    Mat grayL, grayR;
    cvtColor(left, grayL, COLOR_BGR2GRAY);
    cvtColor(right, grayR, COLOR_BGR2GRAY);
    equalizeHist(grayL, grayL);
    equalizeHist(grayR, grayR);

    int win_size = 3;
    int min_disp = 0;
    int num_disp = 64;

    Ptr<StereoSGBM> sgbm = StereoSGBM::create(
        min_disp, num_disp, win_size,
        8 * 3 * win_size * win_size,
        32 * 3 * win_size * win_size,
        1, 15, 50, 16, 63, StereoSGBM::MODE_SGBM_3WAY
    );

    sgbm->compute(grayL, grayR, disparity);
    normalize(disparity, disp_show, 0, 255, NORM_MINMAX, CV_8U);
}

// ====================== 深度计算 ======================
void compute_depth(Mat& disparity, Mat& Q, Mat& depth_heatmap, float& center_depth_out)
{
    Mat dispf, points3D;
    disparity.convertTo(dispf, CV_32F, 1.0 / 16.0);
    reprojectImageTo3D(dispf, points3D, Q);

    Mat z_depth;
    extractChannel(points3D, z_depth, 2);

    int h = z_depth.rows;
    int w = z_depth.cols;
    int cx = w / 2;
    int cy = h / 2;

    float center_depth = z_depth.at<float>(cy, cx);
    if (isinf(center_depth) || isnan(center_depth) || dispf.at<float>(cy,cx) < 0)
        center_depth = 0.0f;
    else
        center_depth /= 500.0f;

    center_depth_out = center_depth;

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

    Mat norm;
    normalize(clipped, norm, 0, 255, NORM_MINMAX, CV_8U);
    Mat reversed = 255 - norm;
    applyColorMap(reversed, depth_heatmap, COLORMAP_HOT);

    char buf[100];
    sprintf(buf, "Depth: %.2fm", center_depth);
    putText(depth_heatmap, buf, Point(50,100), FONT_HERSHEY_SIMPLEX,1,Scalar(0,255,255),2);
    drawMarker(depth_heatmap, Point(cx,cy), Scalar(0,255,255), MARKER_CROSS,20,2);
}

// ====================== 【光流计算 + 绘制方向箭头】 ======================
void draw_optical_flow(Mat& curr_frame, Mat& flow_output)
{
    Mat gray;
    cvtColor(curr_frame, gray, COLOR_BGR2GRAY);
    flow_output = curr_frame.clone();

    // 第一帧初始化
    if (first_frame) {
        prev_gray = gray.clone();
        first_frame = false;
        return;
    }

    // 计算光流
    calcOpticalFlowFarneback(prev_gray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

    // 每隔 20 个像素画一个箭头
    int step = 20;
    for (int y = 0; y < gray.rows; y += step) {
        for (int x = 0; x < gray.cols; x += step) {

            Point2f f = flow.at<Point2f>(y, x);
            float len = sqrt(f.x*f.x + f.y*f.y);

            // 过滤微小运动
            if (len < 0.5) continue;

            Point p(x, y);
            Point to(x + f.x*2, y + f.y*2);

            // 画方向箭头
            arrowedLine(flow_output, p, to, Scalar(0, 255, 0), 1, LINE_AA);
        }
    }

    prev_gray = gray.clone();
}

// ====================== 主函数 ======================
int main()
{
    Mat mapLx, mapLy, mapRx, mapRy, Q;
    stereo_rectify(mapLx, mapLy, mapRx, mapRy, Q);

    VideoCapture cap(1);
    if (!cap.isOpened()) {
        cout << "摄像头打开失败！" << endl;
        return -1;
    }

    cap.set(CAP_PROP_FRAME_WIDTH, 1280*2);
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);

    Mat frame;
    while (waitKey(1) != 'q')
    {
        cap >> frame;
        if (frame.empty()) break;

        // 左右目拆分
        Mat left = frame(Rect(0,0,1280,720));
        Mat right = frame(Rect(1280,0,1280,720));

        // 校正
        Mat Lrect, Rrect;
        remap(left, Lrect, mapLx, mapLy, INTER_LINEAR);
        remap(right, Rrect, mapRx, mapRy, INTER_LINEAR);

        // 深度
        Mat disp, disp_show, depth_map;
        float center_depth;
        compute_disparity(Lrect, Rrect, disp, disp_show);
        compute_depth(disp, Q, depth_map, center_depth);

        // ====================== 【光流】 ======================
        Mat flow_img;
        draw_optical_flow(left, flow_img);

        // 显示窗口
        imshow("Left", left);
        imshow("Depth", depth_map);
        imshow("Optical Flow 运动方向", flow_img);  // <-- 光流窗口
    }

    cap.release();
    destroyAllWindows();
    return 0;
}

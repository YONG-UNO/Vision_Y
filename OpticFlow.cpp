#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
using namespace cv;
using namespace std;

// ==================== 光流全局变量 ====================
Mat prev_gray;
Mat flow;
bool first_frame = true;

// ==================== 计算相机运动方向 ====================
Point2f calculate_camera_motion(Mat &flow)
{
    float sum_x = 0, sum_y = 0;
    int count = 0;
    int step = 20;

    for (int y = 0; y < flow.rows; y += step) {
        for (int x = 0; x < flow.cols; x += step) {
            Point2f f = flow.at<Point2f>(y, x);
            float len = hypot(f.x, f.y);

            if (len < 0.2f) continue;

            sum_x += f.x;
            sum_y += f.y;
            count++;
        }
    }

    if (count == 0) return {0, 0};

    // 平均光流 = 场景运动方向
    Point2f avg_flow(sum_x / count, sum_y / count);

    // 相机运动 = 场景运动 取反！！！（你猜的完全正确）
    Point2f camera_motion = -avg_flow;

    return camera_motion;
}

// ==================== 绘制光流 + 相机方向箭头 ====================
void draw_optical_flow(Mat &frame, Mat &output)
{
    output = frame.clone();
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    if (first_frame) {
        prev_gray = gray.clone();
        first_frame = false;
        return;
    }

    // 计算光流
    calcOpticalFlowFarneback(prev_gray, gray, flow, 0.5, 2, 10, 2, 5, 1.1, 0);

    // 画全场光流（绿色）
    int step = 20;
    for (int y = 0; y < gray.rows; y += step) {
        for (int x = 0; x < gray.cols; x += step) {
            Point2f f = flow.at<Point2f>(y, x);
            if (hypot(f.x, f.y) < 0.2) continue;
            arrowedLine(output, Point(x, y), Point(x + f.x * 2, y + f.y * 2),
                        Scalar(0, 255, 0), 1, LINE_AA);
        }
    }

    // ==================== 相机运动方向（红色大箭头） ====================
    Point2f cam_move = calculate_camera_motion(flow);
    int cx = output.cols / 2;
    int cy = output.rows / 2;

    Point end_point = Point(
        cx + cam_move.x * 50,
        cy + cam_move.y * 50
    );

    arrowedLine(output, Point(cx, cy), end_point, Scalar(0, 0, 255), 3, LINE_AA);

    // 显示文字
    char text[100];
    sprintf(text, "Camera: X=%.1f Y=%.1f", cam_move.x, cam_move.y);
    putText(output, text, Point(30, 60), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);

    prev_gray = gray.clone();
}

// ==================== 主函数（单目） ====================
int main()
{
    VideoCapture cap;
    cap.open(2);   // 单目摄像头（0 或 1）
    if (!cap.isOpened()) {
        cout << "摄像头打开失败！" << endl;
        return -1;
    }

    cap.set(CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);

    Mat frame;
    while (waitKey(1) != 'q') {
        cap >> frame;
        if (frame.empty()) break;

        Mat flow_img;
        draw_optical_flow(frame, flow_img);

        imshow("Optical Flow + Camera Motion", flow_img);
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
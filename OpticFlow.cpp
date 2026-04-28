#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
using namespace cv;
using namespace std;

// ==================== 光流全局 ====================
Mat prev_gray;
Mat flow;
bool first_frame = true;

// ==================== 滤波参数 ====================
Point2f filtered_cam = {0, 0};
const float ALPHA = 0.12f;        // 滤波强度（越小越稳）
const float DEAD_ZONE = 0.07f;    // 静止死区（低于此视为不动）
const float MAX_SPEED = 4.0f;     // 限制最大速度，防止快移跳变

// ==================== 计算相机运动（带异常剔除） ====================
Point2f calculate_camera_motion(Mat &flow)
{
    float sum_x = 0, sum_y = 0;
    int count = 0;
    int step = 20;

    for (int y = 0; y < flow.rows; y += step) {
        for (int x = 0; x < flow.cols; x += step) {
            Point2f f = flow.at<Point2f>(y, x);
            float len = hypot(f.x, f.y);

            if (len < 0.15f) continue;

            sum_x += f.x;
            sum_y += f.y;
            count++;
        }
    }

    if (count < 10) return {0, 0}; // 特征太少直接归零

    Point2f avg_flow(sum_x / count, sum_y / count);
    Point2f cam = -avg_flow;

    // 限制最大速度，防止快移跳变
    float speed = hypot(cam.x, cam.y);
    if (speed > MAX_SPEED) {
        cam.x = cam.x / speed * MAX_SPEED;
        cam.y = cam.y / speed * MAX_SPEED;
    }

    return cam;
}

// ==================== 一阶低通滤波 + 死区防抖 ====================
Point2f smooth_camera_motion(Point2f raw_cam)
{
    // 死区：静止直接归零
    float speed = hypot(raw_cam.x, raw_cam.y);
    if (speed < DEAD_ZONE) {
        filtered_cam = filtered_cam * 0.8f; // 缓慢归零，不抖
        if (hypot(filtered_cam.x, filtered_cam.y) < DEAD_ZONE)
            filtered_cam = {0, 0};
        return filtered_cam;
    }

    // 低通滤波：新值 = 历史*0.88 + 新值*0.12（超级平滑）
    filtered_cam.x = filtered_cam.x * (1 - ALPHA) + raw_cam.x * ALPHA;
    filtered_cam.y = filtered_cam.y * (1 - ALPHA) + raw_cam.y * ALPHA;

    return filtered_cam;
}

// ==================== 工具函数 ====================
float get_angle(Point2f v) {
    float a = atan2(v.y, v.x) * 180.0 / CV_PI;
    return a < 0 ? a + 360 : a;
}

string get_dir(Point2f cam, float speed)
{
    if (speed < DEAD_ZONE) return "STOPPED";
    float ang = get_angle(cam);
    if (ang < 45) return "RIGHT";
    if (ang < 135) return "DOWN";
    if (ang < 225) return "LEFT";
    if (ang < 315) return "UP";
    return "RIGHT";
}

void draw_speed_bar(Mat &img, float speed, int x, int y)
{
    int w = 300, h = 20;
    speed = std::min(speed, 3.0f) / 3.0f;
    rectangle(img, {x, y, w, h}, Scalar(255,255,255), 2);
    rectangle(img, {x, y, (int)(w*speed), h}, Scalar(0,255,255), -1);
}

// ==================== 光流绘制 ====================
void draw_optical_flow(Mat &frame, Mat &output)
{
    output = frame.clone();
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    int cx = output.cols/2, cy = output.rows/2;

    line(output, {cx,0}, {cx,output.rows}, Scalar(255,255,255), 1);
    line(output, {0,cy}, {output.cols,cy}, Scalar(255,255,255), 1);

    if (first_frame) {
        prev_gray = gray.clone();
        first_frame = false;
        return;
    }

    calcOpticalFlowFarneback(prev_gray, gray, flow, 0.5, 2, 10, 2, 5, 1.1, 0);

    // 画光流
    int step = 20;
    for (int y=0; y<gray.rows; y+=step)
        for (int x=0; x<gray.cols; x+=step) {
            Point2f f = flow.at<Point2f>(y,x);
            if (hypot(f.x,f.y) < 0.15) continue;
            arrowedLine(output, {x,y}, {x+f.x*2, y+f.y*2}, {0,255,0}, 1, LINE_AA);
        }

    // ==================== 核心：滤波后运动 ====================
    Point2f raw_cam = calculate_camera_motion(flow);
    Point2f cam_smooth = smooth_camera_motion(raw_cam);
    float speed = hypot(cam_smooth.x, cam_smooth.y);
    string dir = get_dir(cam_smooth, speed);

    // 画相机方向箭头
    Point endp = {cx + cam_smooth.x*70, cy + cam_smooth.y*70};
    arrowedLine(output, {cx,cy}, endp, {0,0,255}, 3, LINE_AA);

    // ==================== 显示面板 ====================
    char buf[100];
    sprintf(buf, "X: %.2f", cam_smooth.x);
    putText(output, buf, {30,60}, 0, 1, {0,255,255}, 2);

    sprintf(buf, "Y: %.2f", cam_smooth.y);
    putText(output, buf, {30,100}, 0, 1, {0,255,255}, 2);

    sprintf(buf, "SPEED: %.2f", speed);
    putText(output, buf, {30,140}, 0, 1, {0,255,255}, 2);

    sprintf(buf, "DIR: %s", dir.c_str());
    putText(output, buf, {30,180}, 0, 1, {0,0,255}, 2);

    draw_speed_bar(output, speed, 30, 230);

    prev_gray = gray.clone();
}

// ==================== 主函数 ====================
int main()
{
    VideoCapture cap(2);
    if (!cap.isOpened()) return -1;

    cap.set(CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);

    Mat frame;
    while (waitKey(1) != 'q') {
        cap >> frame;
        if (frame.empty()) break;

        Mat out;
        draw_optical_flow(frame, out);
        imshow("Optical Flow Stabilized", out);
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
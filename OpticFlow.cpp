#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
using namespace cv;
using namespace std;

// ==================== 全局参数 ====================
Mat prev_gray;
Mat flow;
bool first_frame = true;

// 低通滤波
Point2f filtered_cam = {0, 0};
const float ALPHA      = 0.12f;
const float DEAD_ZONE  = 0.07f;
const float MAX_SPEED  = 4.0f;

// 【无人机运动轨迹】
vector<Point> track_path;
const int TRACK_MAP_W  = 320;
const int TRACK_MAP_H  = 240;
Mat track_map;
float track_scale      = 12.0f;   // 轨迹缩放，调小轨迹变大
Point2f global_pos     = {0, 0}; // 全局累积坐标

// ==================== 光流计算 ====================
Point2f calculate_camera_motion(Mat &flow)
{
    float sum_x = 0, sum_y = 0;
    int count = 0;
    int step = 20;

    for (int y = 0; y < flow.rows; y += step)
    {
        for (int x = 0; x < flow.cols; x += step)
        {
            Point2f f = flow.at<Point2f>(y, x);
            float len = hypot(f.x, f.y);
            if (len < 0.15f) continue;

            sum_x += f.x;
            sum_y += f.y;
            count++;
        }
    }

    if (count < 10) return {0, 0};

    Point2f avg_flow(sum_x / count, sum_y / count);
    Point2f cam = -avg_flow;

    float speed = hypot(cam.x, cam.y);
    if (speed > MAX_SPEED)
    {
        cam.x = cam.x / speed * MAX_SPEED;
        cam.y = cam.y / speed * MAX_SPEED;
    }
    return cam;
}

// ==================== 低通滤波 + 死区 ====================
Point2f smooth_camera_motion(Point2f raw_cam)
{
    float speed = hypot(raw_cam.x, raw_cam.y);
    if (speed < DEAD_ZONE)
    {
        filtered_cam.x *= 0.8f;
        filtered_cam.y *= 0.8f;
        if (hypot(filtered_cam.x, filtered_cam.y) < DEAD_ZONE)
            filtered_cam = {0, 0};
        return filtered_cam;
    }

    filtered_cam.x = filtered_cam.x * (1 - ALPHA) + raw_cam.x * ALPHA;
    filtered_cam.y = filtered_cam.y * (1 - ALPHA) + raw_cam.y * ALPHA;
    return filtered_cam;
}

// ==================== 工具函数 ====================
float get_angle(Point2f v)
{
    float a = atan2(v.y, v.x) * 180.0 / CV_PI;
    return a < 0 ? a + 360 : a;
}

string get_dir(Point2f cam, float speed)
{
    if (speed < DEAD_ZONE) return "STOPPED";
    float ang = get_angle(cam);
    if (ang < 45)      return "RIGHT";
    if (ang < 135)     return "DOWN";
    if (ang < 225)     return "LEFT";
    if (ang < 315)     return "UP";
    return "RIGHT";
}

void draw_speed_bar(Mat &img, float speed, int x, int y)
{
    int w = 300, h = 20;
    speed = min(speed, 3.0f) / 3.0f;
    rectangle(img, Rect(x, y, w, h), Scalar(255,255,255), 2);
    rectangle(img, Rect(x, y, (int)(w*speed), h), Scalar(0,255,255), -1);
}

// ==================== 绘制无人机轨迹 ====================
void update_track(Point2f smooth_move)
{
    // 累积全局位置
    global_pos.x += smooth_move.x;
    global_pos.y += smooth_move.y;

    // 映射到小地图中心
    int mx = TRACK_MAP_W / 2  + global_pos.x * track_scale;
    int my = TRACK_MAP_H / 2  + global_pos.y * track_scale;

    // 限制范围，防止越界
    mx = max(5, min(TRACK_MAP_W - 5, mx));
    my = max(5, min(TRACK_MAP_H - 5, my));

    track_path.emplace_back(mx, my);

    // 限制轨迹最大点数，防止内存暴涨
    if (track_path.size() > 800)
        track_path.erase(track_path.begin());
}

void draw_track_map(Mat &dst)
{
    // 浅灰色背景画布
    track_map = Mat::zeros(TRACK_MAP_H, TRACK_MAP_W, CV_8UC3);
    track_map.setTo(Scalar(30,30,30));

    // 绘制运动轨迹 青色线条
    for (size_t i = 1; i < track_path.size(); i++)
    {
        line(track_map, track_path[i-1], track_path[i], Scalar(0, 200, 255), 1);
    }

    // 原点十字 + 当前无人机红点
    Point center(TRACK_MAP_W/2, TRACK_MAP_H/2);
    drawMarker(track_map, center, Scalar(100,100,100), MARKER_CROSS, 12, 1);
    if (!track_path.empty())
        circle(track_map, track_path.back(), 4, Scalar(0,0,255), -1);

    // 贴到主画面右下角
    Rect roi(dst.cols - TRACK_MAP_W - 10, dst.rows - TRACK_MAP_H - 10, TRACK_MAP_W, TRACK_MAP_H);
    track_map.copyTo(dst(roi));
}

// ==================== 主光流绘制 ====================
void draw_optical_flow(Mat &frame, Mat &output)
{
    output = frame.clone();
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    int cx = output.cols / 2;
    int cy = output.rows / 2;

    line(output, Point(cx, 0), Point(cx, output.rows), Scalar(255,255,255), 1);
    line(output, Point(0, cy), Point(output.cols, cy), Scalar(255,255,255), 1);

    if (first_frame)
    {
        prev_gray = gray.clone();
        first_frame = false;
        track_path.clear();
        global_pos = {0,0};
        return;
    }

    calcOpticalFlowFarneback(prev_gray, gray, flow, 0.5, 2, 10, 2, 5, 1.1, 0);

    // 绘制稀疏绿色光流
    int step = 20;
    for (int y = 0; y < gray.rows; y += step)
    {
        for (int x = 0; x < gray.cols; x += step)
        {
            Point2f f = flow.at<Point2f>(y, x);
            if (hypot(f.x, f.y) < 0.15) continue;
            arrowedLine(output, Point(x,y), Point(x+f.x*2, y+f.y*2), Scalar(0,255,0), 1, LINE_AA);
        }
    }

    // 滤波平滑运动
    Point2f raw_cam  = calculate_camera_motion(flow);
    Point2f cam_smooth = smooth_camera_motion(raw_cam);
    float speed = hypot(cam_smooth.x, cam_smooth.y);
    string dir  = get_dir(cam_smooth, speed);

    // 更新+绘制飞行轨迹
    update_track(cam_smooth);
    draw_track_map(output);

    // 中心红色运动箭头
    Point endp = Point(cx + cam_smooth.x*70, cy + cam_smooth.y*70);
    arrowedLine(output, Point(cx,cy), endp, Scalar(0,0,255), 3, LINE_AA);

    // 文本信息
    char buf[100];
    sprintf(buf, "X: %.2f", cam_smooth.x);
    putText(output, buf, Point(30,60), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,255), 2);

    sprintf(buf, "Y: %.2f", cam_smooth.y);
    putText(output, buf, Point(30,100), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,255), 2);

    sprintf(buf, "SPEED: %.2f", speed);
    putText(output, buf, Point(30,140), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,255), 2);

    sprintf(buf, "DIR: %s", dir.c_str());
    putText(output, buf, Point(30,180), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255), 2);

    draw_speed_bar(output, speed, 30, 230);

    prev_gray = gray.clone();
}

// ==================== 主函数 ====================
int main()
{
    VideoCapture cap(2);
    if (!cap.isOpened())
    {
        cout << "摄像头打开失败" << endl;
        return -1;
    }

    cap.set(CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);

    Mat frame;
    while (waitKey(1) != 'q')
    {
        cap >> frame;
        if (frame.empty()) break;

        Mat out;
        draw_optical_flow(frame, out);
        imshow("Drone Optical Flow + Track", out);
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
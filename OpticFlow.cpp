#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace cv;
using namespace std;

// ==================== 全局参数 ====================
Mat prev_gray;
Mat flow;
bool first_frame = true;

// 滤波防抖
Point2f filtered_cam = {0, 0};
const float ALPHA      = 0.12f;
const float DEAD_ZONE  = 0.07f;
const float MAX_SPEED  = 4.0f;

// 无人机全局位置 + 轨迹
vector<Point2f> track_path;
Point2f global_pos = {0, 0};

// 轨迹画布配置
const int MAP_W = 320;
const int MAP_H = 240;
float view_scale = 15.0f;     // 基础缩放
const float MIN_SCALE = 3.0f;
const float MAX_SCALE = 40.0f;

// ==================== 光流运动计算 ====================
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

// 低通滤波 + 死区
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

// 工具
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

// ==================== 核心：自动缩放轨迹 ====================
void update_global_pos(Point2f move)
{
    global_pos += move;
    track_path.push_back(global_pos);
    // 限制轨迹长度，防止无限堆积
    if(track_path.size() > 1200)
        track_path.erase(track_path.begin());
}

void draw_auto_scale_track(Mat &dst)
{
    Mat map = Mat::zeros(MAP_H, MAP_W, CV_8UC3);
    map.setTo(Scalar(25,25,25));

    if(track_path.empty()) return;

    // 1. 取出所有轨迹点，计算包围盒
    float minX = track_path[0].x, maxX = track_path[0].x;
    float minY = track_path[0].y, maxY = track_path[0].y;
    for(auto &p : track_path)
    {
        minX = min(minX, p.x);
        maxX = max(maxX, p.x);
        minY = min(minY, p.y);
        maxY = max(maxY, p.y);
    }

    // 2. 自动计算需要的缩放比例
    float rangeX = maxX - minX;
    float rangeY = maxY - minY;
    float rangeMax = max(rangeX, rangeY);

    // 动态自适应缩放
    if(rangeMax > 0.1f)
    {
        view_scale = (MAP_W * 0.45f) / rangeMax;
        view_scale = clamp(view_scale, MIN_SCALE, MAX_SCALE);
    }

    // 3. 中心点偏移，让轨迹居中
    float centerX = (minX + maxX) * 0.5f;
    float centerY = (minY + maxY) * 0.5f;
    int mapCx = MAP_W / 2;
    int mapCy = MAP_H / 2;

    // 4. 转换坐标绘制轨迹
    vector<Point> pts;
    for(auto &p : track_path)
    {
        int px = mapCx + (p.x - centerX) * view_scale;
        int py = mapCy + (p.y - centerY) * view_scale;
        pts.emplace_back(px, py);
    }

    // 画轨迹线
    for(int i = 1; i < pts.size(); i++)
    {
        line(map, pts[i-1], pts[i], Scalar(0, 210, 255), 1);
    }

    // 原点十字
    Point2f origin;
    int ox = mapCx + (origin.x - centerX) * view_scale;
    int oy = mapCy + (origin.y - centerY) * view_scale;
    drawMarker(map, Point(ox, oy), Scalar(80,80,80), MARKER_CROSS, 10, 1);

    // 当前无人机红点
    circle(map, pts.back(), 5, Scalar(0,0,255), -1);

    // 贴到画面右下角
    Rect roi(dst.cols - MAP_W - 10, dst.rows - MAP_H - 10, MAP_W, MAP_H);
    map.copyTo(dst(roi));
}

// ==================== 光流主绘制 ====================
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
        view_scale = 15.0f;
        return;
    }

    calcOpticalFlowFarneback(prev_gray, gray, flow, 0.5, 2, 10, 2, 5, 1.1, 0);

    // 绿色稀疏光流
    int step = 20;
    for (int y = 0; y < gray.rows; y += step)
    {
        for (int x = 0; x < gray.cols; x += step)
        {
            Point2f f = flow.at<Point2f>(y, x);
            if (hypot(f.x, f.y) < 0.15f) continue;
            arrowedLine(output, Point(x,y), Point(x+f.x*2, y+f.y*2), Scalar(0,255,0), 1, LINE_AA);
        }
    }

    Point2f raw_cam = calculate_camera_motion(flow);
    Point2f cam_smooth = smooth_camera_motion(raw_cam);
    float speed = hypot(cam_smooth.x, cam_smooth.y);
    string dir  = get_dir(cam_smooth, speed);

    // 更新位置 + 自动缩放轨迹
    update_global_pos(cam_smooth);
    draw_auto_scale_track(output);

    // 中心运动箭头
    Point endp = Point(cx + cam_smooth.x*70, cy + cam_smooth.y*70);
    arrowedLine(output, Point(cx,cy), endp, Scalar(0,0,255), 3, LINE_AA);

    // 信息文本
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
        imshow("Drone Optical Flow | Auto Scale Track", out);
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
#include <opencv4/opencv2/opencv.hpp>

cv::Mat K_left = (cv::Mat_<double>(3,3) <<
    782.131, 0, 629.4724,
    0, 780.0977, 371.0506,
    0, 0, 1.0000);

cv::Mat K_right = (cv::Mat_<double>(3,3) <<
    781.1365, 0, 595.5944,
    0, 779.0337, 371.4303,
    0, 0, 1.0000);

int main() {
    cv::VideoCapture cap(1);

    cv::Mat frame; // 存放每一帧图片

    for (;;) {
        cap.read(frame);
        cv::imshow("",frame);
        if (cv::waitKey(1) == 'q')
            break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
//
// Created by wjg on 18-1-22.
//

#ifndef ORB_SLAM2_BASEDETECTOR_H
#define ORB_SLAM2_BASEDETECTOR_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


class BaseDetector {
public:
    virtual void Run() = 0;
    virtual void Stop() = 0;
    virtual cv::Mat GetDetection() = 0;
    virtual cv::Mat GetOverlay() = 0;

    volatile bool newImageArrived;
    cv::Mat newImage;
};

extern BaseDetector* globalDetector;

#endif //ORB_SLAM2_BASEDETECTOR_H

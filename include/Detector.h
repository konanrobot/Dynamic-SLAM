//
// Created by wjg on 17-5-15.
//

#ifndef ORB_SLAM2_DETECTOR_H
#define ORB_SLAM2_DETECTOR_H


#include <caffe/caffe.hpp>
#include "BaseDetector.h"

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <mutex>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)

struct Detection;

class Detector : public BaseDetector {
public:
    Detector(const string& model_file,
             const string& weights_file,
             const string& mean_file,
             const string& mean_value);

    virtual void Run();

    std::vector<vector<float> > Detect(const cv::Mat& img);
    virtual void Stop();
    virtual cv::Mat GetDetection();
    virtual cv::Mat GetOverlay();

//    volatile bool newImageArrived;
//    cv::Mat newImage;

private:
    void SetMean(const string& mean_file, const string& mean_value);

    void WrapInputLayer(std::vector<cv::Mat>* input_channels);

    void Preprocess(const cv::Mat& img,
                    std::vector<cv::Mat>* input_channels);

private:
    boost::shared_ptr<Net<float> > net_;
    cv::Size image_size;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
    std::mutex mMutex;
    volatile bool bStop;
    const float confidence_threshold = 0.6;
    std::vector<Detection> currentDetections;  //Detected objects in current keyframe
    std::vector<Detection> lastDetections;      //Detected objects in last keyframe
    std::set<int> dynamicTypes;
};

struct Detection {
    // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
    float image_id;
    float label;
    float score;
    float xmin;
    float ymin;
    float xmax;
    float ymax;
};

//string mean_file = "";
//string mean_value = "104,117,123";

//DEFINE_string(mean_file, "",
//              "The mean file used to subtract from the input image.");
//DEFINE_string(mean_value, "104,117,123",
//              "If specified, can be one value or can be same as image channels"
//                      " - would subtract from the corresponding channel). Separated by ','."
//                      "Either mean_file or mean_value should be provided, not both.");
//DEFINE_string(file_type, "image",
//              "The file type in the list_file. Currently support image and video.");
//DEFINE_string(out_file, "",
//              "If provided, store the detection results in the out_file.");
//DEFINE_double(confidence_threshold, 0.01,
//              "Only store detections with score higher than the threshold.");

//extern Detector* globalDetector;

#endif  // USE_OPENCV

#endif //ORB_SLAM2_DETECTOR_H

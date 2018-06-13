//
// Created by wjg on 18-1-22.
//

#ifndef ORB_SLAM2_SEGMENTOR_H
#define ORB_SLAM2_SEGMENTOR_H

#include "BaseDetector.h"

#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <chrono> //Just for time measurement

using namespace caffe;  // NOLINT(build/namespaces)

class Segmentor : public BaseDetector {

public:
    Segmentor(const string& model_file,
               const string& trained_file, const string& LUT_file);
    virtual void Run();
    virtual void Stop();
    virtual cv::Mat GetDetection();
    virtual cv::Mat GetOverlay();

private:
    void Predict(const cv::Mat& img, string LUT_file);

    void SetMean(const string& mean_file);

    void WrapInputLayer(std::vector<cv::Mat>* input_channels);

    void Preprocess(const cv::Mat& img,
                    std::vector<cv::Mat>* input_channels);

    void Visualization(Blob<float>* output_layer, string LUT_file);

private:
    std::shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    std::mutex mMutex;
    string LUT_file;
    volatile bool bStop;
    cv::Size input_size;

    cv::Mat currentDetection;

};


#endif //ORB_SLAM2_SEGMENTOR_H

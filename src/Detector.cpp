//
// Created by wjg on 17-5-15.
//

#include "Detector.h"
#include "Label.h"

#ifdef USE_OPENCV

//Detector* globalDetector;

Detector::Detector(const string& model_file,
                   const string& weights_file,
                   const string& mean_file,
                   const string& mean_value) : BaseDetector()  {

    newImageArrived = false;
    bStop = false;
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif

    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(weights_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
            << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    /* Load the binaryproto mean file. */
    SetMean(mean_file, mean_value);

    dynamicTypes = {2, 6, 7, 15};
}

void Detector::Run() {
    Caffe::set_mode(Caffe::GPU);
    while (!bStop)
    {
        if (newImageArrived)
        {
            //TODO: TEST the time used!!!!!!!!!
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            Detect(newImage);
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
            double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
//            std::cout << "Detection used " << std::fixed << ttrack << "seconds." << std::endl;
            newImageArrived = false;
        }
//        usleep(5000);
    }
    std::cout << "Detector finished!!!" << std::endl;
}

void Detector::Stop() {
    bStop = true;
}

std::vector<vector<float> > Detector::Detect(const cv::Mat& img) {
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    // Update image size
    image_size.height = img.rows;
    image_size.width = img.cols;

    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
                         input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);

    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

    Preprocess(img, &input_channels);

    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();

    net_->Forward();

    std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();

    /* Copy the output layer to a std::vector */
    Blob<float>* result_blob = net_->output_blobs()[0];
    const float* result = result_blob->cpu_data();
    const int num_det = result_blob->height();
    vector<vector<float> > detections;
    for (int k = 0; k < num_det; ++k) {
        if (result[0] == -1) {
            // Skip invalid detection.
            result += 7;
            continue;
        }
        vector<float> detection(result, result + 7);
        detections.push_back(detection);
        result += 7;
    }

    std::chrono::steady_clock::time_point t6 = std::chrono::steady_clock::now();

    //Copy variables within scoped mutex
    {
        std::unique_lock<std::mutex> lock(mMutex);
        currentDetections.clear();
        for (int i = 0; i < detections.size(); ++i)
        {
            // Save only detections exceeding confidence threshold
            if (detections[i][2] < confidence_threshold) {
                continue;
            }
            //Save only detections of dynamic types
            if (dynamicTypes.find(detections[i][1]) == dynamicTypes.end())
            {
                continue;
            }
            // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
            Detection d;
            d.image_id = detections[i][0];
            d.label = detections[i][1];
            d.score = detections[i][2];
            d.xmin = detections[i][3] * img.cols;
            d.ymin = detections[i][4] * img.rows;
            d.xmax = detections[i][5] * img.cols;
            d.ymax = detections[i][6] * img.rows;
            currentDetections.push_back(d);
//            std::cout << "[" << d.image_id << "," << d.label << "," << d.score << "," << d.xmin << "," << d.ymin << "," << d.xmax << "," << d.ymax << "]" << std::endl;
        }

        // Add extra car detection manually. Only for momenta_multiloop dataset.
//        Detection d;
//        d.label = 7;
//        d.score = 0.99;
//        d.xmin = 0;
//        d.ymin = 352;
//        d.xmax = 639;
//        d.ymax = 479;
//        currentDetections.push_back(d);

        // recover the missed detections from last frame
//        for (int i=0; i<lastDetections.size(); i++)
//        {
//            bool missed = true;
//            for (int j=0; j<currentDetections.size(); j++)
//            {
//                float xmin0 = lastDetections[i].xmin;
//                float ymin0 = lastDetections[i].ymin;
//                float xmax0 = lastDetections[i].xmax;
//                float ymax0 = lastDetections[i].ymax;
//                float xmin1 = currentDetections[j].xmin;
//                float ymin1 = currentDetections[j].ymin;
//                float xmax1 = currentDetections[j].xmax;
//                float ymax1 = currentDetections[j].ymax;
//                float distance = abs(xmin0 - xmin1) + abs(ymin0 - ymin1) + abs(xmax0 - xmax1) + abs(ymax0 - ymax1);
////                std::cout << "Distance between lastDetections[" << i << "] and currentDetection[" << j << "] is " << distance << std::endl;
//                if (distance < 250)
//                {
//                    missed = false;
//                    break;
//                }
//            }
//            if (missed)
//            {
//                currentDetections.push_back(lastDetections[i]);
//            }
//        }
//        lastDetections.clear();
//        lastDetections.assign(currentDetections.cbegin(), currentDetections.cend());

    } // destroy scoped mutex -> release mutex

    std::chrono::steady_clock::time_point t7 = std::chrono::steady_clock::now();

    double ttrack1= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    double ttrack2= std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t2).count();
    double ttrack3= std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3).count();
    double ttrack4= std::chrono::duration_cast<std::chrono::duration<double> >(t5 - t4).count();
    double ttrack5= std::chrono::duration_cast<std::chrono::duration<double> >(t6 - t5).count();
    double ttrack6= std::chrono::duration_cast<std::chrono::duration<double> >(t7 - t6).count();
//    std::cout << "[" << std::fixed << ttrack1 << " , " << ttrack2 << " , " << ttrack3 << " , " << ttrack4 << " , " << ttrack5 << " , " << ttrack6 << "]" << std::endl;

    return detections;
}

cv::Mat Detector::GetDetection()
{
    if (currentDetections.empty())
    {
        return cv::Mat();
    }

    cv::Mat detection = cv::Mat::zeros(image_size.height, image_size.width, CV_8U);

    //Copy variables within scoped mutex
    {
        std::unique_lock<std::mutex> lock(mMutex);
        for (int i = 0; i < currentDetections.size(); ++i) {
            if (dynamicTypes.find(currentDetections[i].label) == dynamicTypes.end()) {
                continue;
            }
            for (int j = currentDetections[i].ymin; j < currentDetections[i].ymax; j++) {
                for (int k = currentDetections[i].xmin; k < currentDetections[i].xmax; k++) {
                    if (j < 0 || j >= detection.rows || k < 0 || k > detection.cols)
                    {
                        continue;
                    }
                    detection.at<uchar>(j, k) = 1;
                }
            }
        }
    }
    return detection;
}

cv::Mat Detector::GetOverlay() {
    if (currentDetections.empty())
    {
        return cv::Mat();
    }
    cv::Mat overlay = cv::Mat::zeros(image_size.height, image_size.width, CV_8UC3);
    for (int i = 0; i < currentDetections.size(); ++i)
    {
        Detection& d = currentDetections[i];
//      std::cout << "draw a rectangle = [" << d.xmin << "," << d.ymin << "," << d.xmax << "," << d.ymax << "]" << std::endl;
        cv::Point2f pt1,pt2;
        pt1.x=d.xmin;
        pt1.y=d.ymin;
        pt2.x=d.xmax;
        pt2.y=d.ymax;
        cv::rectangle(overlay,pt1,pt2,cv::Scalar(0,0,255), CV_FILLED);
        cv::Point2f pt3;
        pt3.x = d.xmin;
        pt3.y = d.ymax;
        cv::putText(overlay, Label::getLabelByID(d.label), pt3, cv::FONT_HERSHEY_PLAIN,2,cv::Scalar(0,0,255), 2);
    }
//    cv::rectangle(overlay, cv::Point2f(0, 0), cv::Point2f(image_size.width, image_size.height), cv::Scalar(0, 0, 255), CV_FILLED);
//    cout << "image_size = (" << image_size.height << ", " << image_size.width << ")" << endl;
    return overlay;
}

/* Load the mean file in binaryproto format. */
void Detector::SetMean(const string& mean_file, const string& mean_value) {
    cv::Scalar channel_mean;
    if (!mean_file.empty()) {
        CHECK(mean_value.empty()) <<
                                  "Cannot specify mean_file and mean_value at the same time";
        BlobProto blob_proto;
        ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

        /* Convert from BlobProto to Blob<float> */
        Blob<float> mean_blob;
        mean_blob.FromProto(blob_proto);
        CHECK_EQ(mean_blob.channels(), num_channels_)
                << "Number of channels of mean file doesn't match input layer.";

        /* The format of the mean file is planar 32-bit float BGR or grayscale. */
        std::vector<cv::Mat> channels;
        float* data = mean_blob.mutable_cpu_data();
        for (int i = 0; i < num_channels_; ++i) {
            /* Extract an individual channel. */
            cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
            channels.push_back(channel);
            data += mean_blob.height() * mean_blob.width();
        }

        /* Merge the separate channels into a single image. */
        cv::Mat mean;
        cv::merge(channels, mean);

        /* Compute the global mean pixel value and create a mean image
         * filled with this value. */
        channel_mean = cv::mean(mean);
        mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
    }
    if (!mean_value.empty()) {
        CHECK(mean_file.empty()) <<
                                 "Cannot specify mean_file and mean_value at the same time";
        stringstream ss(mean_value);
        vector<float> values;
        string item;
        while (getline(ss, item, ',')) {
            float value = std::atof(item.c_str());
            values.push_back(value);
        }
        CHECK(values.size() == 1 || values.size() == num_channels_) <<
                                                                    "Specify either 1 mean_value or as many as channels: " << num_channels_;

        std::vector<cv::Mat> channels;
        for (int i = 0; i < num_channels_; ++i) {
            /* Extract an individual channel. */
            cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
                            cv::Scalar(values[i]));
            channels.push_back(channel);
        }
        cv::merge(channels, mean_);
    }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void Detector::Preprocess(const cv::Mat& img,
                          std::vector<cv::Mat>* input_channels) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == net_->input_blobs()[0]->cpu_data())
            << "Input channels are not wrapping the input layer of the network.";
}

#endif  // USE_OPENCV
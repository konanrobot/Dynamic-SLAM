//
// Created by wjg on 17-5-23.
//


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>
#include<Converter.h>
#include<Detector.h>

using namespace std;
using namespace cv;

string folderName = "rgb_dataset_human_static02";

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

Mat mK;
Mat mDistCoef;

void getDistortMat(const string& settingFilePath);

int main(int argc, char **argv)
{
    if(argc != 6)
    {
        cerr << endl << "Usage: ./mono_webcam path_to_vocabulary path_to_settings model_file weights_file" << endl;
        return 1;
    }

    VideoCapture cap(0);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    double width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    double height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    cout << "(" << width << " x " << height << ")" << endl;
    if(!cap.isOpened())
    {
        cout << "Open camera failed!" << endl;
        return -1;
    }
    Mat frame;


    getDistortMat(argv[2]);

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string strFile = string(argv[3])+"/rgb.txt";

//    int nImages = vstrImageFilenames.size();

    stringstream ss;
    ofstream f;
    f.open("AllTrajectory.txt");
    ofstream ofs("/home/wjg/datasets/" + folderName + "/rgb.txt");

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,true);

    // Create Object Detection system. It loads the models and gets ready to process frames.
    const string& model_file = argv[4];
    const string& weights_file = argv[5];
    string mean_file = "";
    string mean_value = "104,117,123";
    Detector* detector;
    detector = new Detector(model_file, weights_file, mean_file, mean_value);
    globalDetector = detector;
//    SLAM.setDetector(detector);
    std::thread* thread_detector = new thread(&Detector::Run, detector);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
//    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << mK << endl;
    cout << mDistCoef << endl;
    // Main loop
    cv::Mat im;
    stringstream tframess;
    for(int ni=0; ; ni++)
    {
        cap>>frame;
        Mat temp = frame.clone();

        cv::undistort(temp, frame, mK, mDistCoef);

        struct timeval tv;
        gettimeofday(&tv, 0);
        double tframe = tv.tv_sec + tv.tv_usec / 1000000.0;
        // Read image from file
//        im = cv::imread(string(argv[3])+"/"+vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
        im = frame;
//        double tframe = vTimestamps[ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenames[ni] << endl;
            return 1;
        }

        tframess << fixed << tframe;
        bool success = cv::imwrite("/home/wjg/datasets/" + folderName + "/rgb/" + tframess.str() + ".png", frame);
        if (!success)
        {
            cout << "Save failed!" << endl;
        }
        tframess << " rgb/" << fixed << tframe << ".png" << endl;
        ofs << tframess.str();
        tframess.str("");
        tframess.clear();

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        if (ni % 2 == 0 && !detector->newImageArrived)
        {
//            detector->Detect(im);
            cv::Mat imClone = im.clone();
            detector->newImage = imClone;
//            detector->newImageArrived = true;
        }

        // Pass the image to the SLAM system
        cv::Mat result = SLAM.TrackMonocular(im,tframe);

//        cout << result << endl;
        if (!result.empty())
        {
            vector<float> q = ORB_SLAM2::Converter::toQuaternion(result);
            ss << fixed << tframe << " " << result.at<float>(0, 3) << " " << result.at<float>(1, 3) << " " << result.at<float>(2, 3) << " " << q.at(0) << " " << q.at(1) << " " << q.at(2) << " " << q.at(3) << endl;
            f << ss.str();
            ss.str("");
            ss.clear();
        }




#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack.push_back(ttrack);

        // Wait to load the next frame
//        double T=0;
//        if(ni<nImages-1)
//            T = vTimestamps[ni+1]-tframe;
//        else if(ni>0)
//            T = tframe-vTimestamps[ni-1];
//
//        if(ttrack<T)
//            usleep((T-ttrack)*1e6);
    }

    f.close();

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    int nImages = vTimesTrack.size();
    for(int ni=0; ni<vTimesTrack.size(); ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}

void getDistortMat(const string& settingFilePath)
{
    //Check settings file
    cv::FileStorage fSettings(settingFilePath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];
    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);
    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);
}
/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>

#include <System.h>
#include <Converter.h>
#include <Detector.h>
#include <Segmentor.h>
#include <BaseDetector.h>
#include <DataStatistics.h>

using namespace std;

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc < 6)
    {
        cerr << endl << "Usage: ./mono_tum path_to_vocabulary path_to_settings path_to_sequence model_file weights_file [LUT_file]" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string strFile = string(argv[3])+"/rgb.txt";
    LoadImages(strFile, vstrImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();

    /********************************/
    /******** Added by jinge ********/
    /********************************/
    stringstream ss;
    ofstream f;
    f.open("AllTrajectory.txt");
    ofstream f_kps_mps;
    f_kps_mps.open("kps_mps.txt");

    // Create Object Detection system. It loads the models and gets ready to process frames.
    const string& model_file = argv[4];
    const string& weights_file = argv[5];
    string LUT_file;
    if (argc > 6)
    {
        LUT_file = argv[6];
    }
    
    string mean_file = "";
    string mean_value = "104,117,123";
    globalDetector = new Detector(model_file, weights_file, mean_file, mean_value);
//    globalDetector = new Segmentor(model_file, weights_file, LUT_file);
    thread detectThread(&BaseDetector::Run, globalDetector);
    /********************************/
    /************* End **************/
    /********************************/
    
    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat im;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image from file
        im = cv::imread(string(argv[3])+"/"+vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        /********************************/
        /******** Added by jinge ********/
        /********************************/
//        // flip 180 degrees
//        cout << "im.size = (" << im.cols << ", " << im.rows << ")" << endl;
//        cv::Mat newIm(im.rows, im.cols, im.type());
//        cv::flip(im, newIm, -1);
//        im = newIm;
        /********************************/
        /************* End **************/
        /********************************/

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenames[ni] << endl;
            return 1;
        }

        /********************************/
        /******** Added by jinge ********/
        /********************************/
        globalDetector->newImage = im.clone();
        globalDetector->newImageArrived = true;
        /********************************/
        /************* End **************/
        /********************************/

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        /********************************/
        /******* Modified by jinge ******/
        /********************************/
        // Pass the image to the SLAM system
        cv::Mat result = SLAM.TrackMonocular(im,tframe);
//        cout << result << endl;
        if (!result.empty())
        {
            vector<float> q = ORB_SLAM2::Converter::toQuaternion(result);
            ss << fixed << vTimestamps[ni] << " " << result.at<float>(0, 3) << " " << result.at<float>(1, 3) << " " << result.at<float>(2, 3) << " " << q.at(0) << " " << q.at(1) << " " << q.at(2) << " " << q.at(3) << endl;
            f << ss.str();
            ss.str("");
            ss.clear();
        }

        //统计每一帧时的kps和mps
//        int nKFs = SLAM.mpMap->KeyFramesInMap();
//        int nMPs = SLAM.mpMap->MapPointsInMap();
//        ss << nKFs << " " << nMPs << endl;
//        f_kps_mps << ss.str();
//        ss.str("");
//        ss.clear();
        /********************************/
        /************* End **************/
        /********************************/

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }
    // Stop all threads
    SLAM.Shutdown();

    /********************************/
    /******** Added by jinge ********/
    /********************************/
    f.close();
    f_kps_mps.close();
    globalDetector->Stop();
    detectThread.join();
    /********************************/
    /************* End **************/
    /********************************/

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;
    cout << "total tracking time:" << totaltime << endl;

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    DataStatistics::SaveKeyPointNumbers("KeyPointNumbers.txt");


    return 0;
}

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());

    // skip first three lines
    string s0;
    getline(f,s0);
    getline(f,s0);
    getline(f,s0);

    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenames.push_back(sRGB);
        }
    }
}

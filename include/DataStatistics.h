//
// Created by wjg on 18-6-13.
//

#ifndef ORB_SLAM2_DATASTATISTICS_H
#define ORB_SLAM2_DATASTATISTICS_H

#include <vector>
#include <string>

class DataStatistics {

public:
    static std::vector<int> allKeyPointNums;
    static std::vector<int> filteredKeyPointNums;

    static void SaveKeyPointNumbers(const std::string& filename);

};

#endif //ORB_SLAM2_DATASTATISTICS_H

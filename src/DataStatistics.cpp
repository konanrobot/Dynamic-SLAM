//
// Created by wjg on 18-6-13.
//

#include <ios>
#include <fstream>
#include <iostream>
#include "DataStatistics.h"

using namespace std;

std::vector<int> DataStatistics::allKeyPointNums;

std::vector<int> DataStatistics::filteredKeyPointNums;

void DataStatistics::SaveKeyPointNumbers(const string& filename) {

    ofstream f;
    f.open(filename.c_str());
    f << fixed;


    for (size_t i=0; i<allKeyPointNums.size(); i++)
    {
        f << filteredKeyPointNums[i] << "\t" << allKeyPointNums[i] << endl;
    }

    f.close();
    cout << "Key point numbers saved!" << endl;

}
//
// Created by wjg on 17-5-15.
//

#ifndef ORB_SLAM2_LABEL_H
#define ORB_SLAM2_LABEL_H

#include <string>

using namespace std;

class Label {

public:
    static string getLabelByID(int id);

private:
    static string labels[];
};





#endif //ORB_SLAM2_LABEL_H

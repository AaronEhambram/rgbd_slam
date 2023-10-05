#ifndef LANDMARK_H_
#define LANDMARK_H_
#include "rgbd_slam/Frame.hpp"
#include "rgbd_slam/Observation.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include "Eigen/Dense"
#include <map>
#include <opencv2/imgproc/imgproc.hpp>

class Observation;
class Frame;
class Landmark
{
  public:
  std::map<Frame*,Observation*> obsv_map_;
  cv::Scalar color_;
  bool good_ = true; 
};

#endif
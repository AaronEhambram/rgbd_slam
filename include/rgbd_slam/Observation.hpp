#ifndef OBSERVATION_H_
#define OBSERVATION_H_
#include "rgbd_slam/Frame.hpp"
#include "rgbd_slam/Landmark.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include "Eigen/Dense"
#include <map>
#include <opencv2/imgproc/imgproc.hpp>

class Landmark;
class Frame;
class Observation
{
  public:
  cv::Point2f f_;
  Eigen::Vector3d lcam_point_; 
  Frame* frame_;
  Landmark* landmark_;
};

#endif
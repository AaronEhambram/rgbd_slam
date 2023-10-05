#ifndef FRAME_H_
#define FRAME_H_
#include "rgbd_slam/Observation.hpp"
#include "rgbd_slam/Landmark.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include "Eigen/Dense"
#include <map>
#include <opencv2/imgproc/imgproc.hpp>
#include <mutex>

class Landmark;
class Observation;
class Frame
{
  public:
  double timestamp_;
  std::vector<Landmark*> seen_landmarks_;
  bool is_keyframe_ = false;
  Eigen::Affine3d cam_prev_T_cam_cur_ = Eigen::Affine3d::Identity();
  std::mutex mutex_; 
};

#endif
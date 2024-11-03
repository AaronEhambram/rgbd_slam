#ifndef RGBDSLAM_TYPES_H_
#define RGBDSLAM_TYPES_H_
#include <opencv2/core.hpp>
#include "Eigen/Dense"

struct CalibartionParameters
{
  cv::Mat rgb_cam;
  cv::Mat rgb_distor;
  cv::Mat depth_cam;
  cv::Mat depth_distor;
  Eigen::Affine3d rgb_T_depth;
};

struct RGBDSLAMParameters
{
  int num_horizontal_tiles = 4;
  int num_vertical_tiles = 3;
  int im_width;
  int im_height;
  double max_optical_flow_error;
  int min_features_in_mask_to_detect_new;
};

#endif

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

inline Eigen::Vector<double, 6> ToVector(const Eigen::Affine3d &pose_affine)
{
  const Eigen::AngleAxisd aa(pose_affine.linear());
  const Eigen::Vector3d aa_rotation = aa.axis() * aa.angle();
  Eigen::Vector<double, 6> pose_vector;
  pose_vector.head<3>() = pose_affine.translation();
  pose_vector.tail<3>() = aa_rotation;
  return pose_vector;
}

inline Eigen::Affine3d ToAffine(const Eigen::Vector<double, 6> &pose_vector)
{
  const double angle = pose_vector.tail<3>().norm();
  const Eigen::Vector3d axis = pose_vector.tail<3>().normalized();
  Eigen::AngleAxisd aa{angle, axis};
  Eigen::Affine3d pose_affine{};
  pose_affine.translation() = pose_vector.head<3>();
  pose_affine.linear() = aa.toRotationMatrix();
  return pose_affine;
}

#endif

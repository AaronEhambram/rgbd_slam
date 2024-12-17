#ifndef DEPTHIMAGEODOMETRY_H_
#define DEPTHIMAGEODOMETRY_H_
#include "Eigen/Dense"
#include <thread>
#include <tuple>
#include <optional>
#include "types.hpp"

class DepthImageOdometry
{
public:
  DepthImageOdometry(const CalibartionParameters &calibration_params);
  void Track(const double &timestamp, const cv::Mat &rgb_im, const cv::Mat &depth_im);

private:
  CalibartionParameters calibration_params_;
  double DetermineDepthError(const cv::Mat &prev_depth_im, const cv::Mat &cur_depth_im, const Eigen::Isometry3d &prev_T_cur) const;
  Eigen::Matrix<double, 1, 6> NumericJacobian(const cv::Mat &prev_depth_im, const cv::Mat &cur_depth_im, const Eigen::Isometry3d &prev_T_cur) const;
  Eigen::Vector<double, 6> ToPoseVector(const Eigen::Isometry3d &pose_isometry) const;
  Eigen::Isometry3d ToIsometry(const Eigen::Vector<double, 6> &pose_vector) const;
  Eigen::Isometry3d OptimizePose(const cv::Mat &prev_depth_im, const cv::Mat &cur_depth_im, const Eigen::Isometry3d &prev_T_cur) const;
  double InterpolateDepth(const cv::Mat &depth_im, const double &u, const double &v) const;
  cv::Mat prev_depth_im_;
  Eigen::Isometry3d start_T_cur_;
};

#endif
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

  cv::Mat prev_depth_im_;
};

#endif
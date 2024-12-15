#include "rgbd_slam/DepthImageOdometry.hpp"
#include <opencv2/highgui.hpp>

DepthImageOdometry::DepthImageOdometry(const CalibartionParameters &calibration_params) : calibration_params_{calibration_params}
{
}

void DepthImageOdometry::Track(const double &timestamp, const cv::Mat &rgb_im, const cv::Mat &depth_im)
{
  // Visualize
  cv::Mat depth_im_scaled;
  cv::Mat diff;
  if (prev_depth_im_.size() == depth_im.size())
  {
    cv::absdiff(depth_im, prev_depth_im_, diff);
    diff.convertTo(depth_im_scaled, CV_32F, 0.0001, 0);
    cv::imshow("depth image", depth_im_scaled);
    cv::waitKey(1);
  }

  // Last operations
  depth_im.copyTo(prev_depth_im_);
}
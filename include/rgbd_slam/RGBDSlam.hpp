#ifndef RGBDSLAM_H_
#define RGBDSLAM_H_
#include "Eigen/Dense"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <thread>
#include "rgbd_slam/types.hpp"

class RGBDSlam
{
public:
  RGBDSlam(const CalibartionParameters &calibration_params, const RGBDSLAMParameters &slam_params);
  void Track(const double &timestamp, const cv::Mat &rgb_im, const cv::Mat &depth_im);

private:
  CalibartionParameters calibration_params_;
  RGBDSLAMParameters slam_params_;

  // feature detection and tracking:
  cv::cuda::GpuMat rgb_im_, rgb_im_prev_, depth_im_;
  cv::Ptr<cv::cuda::ORB> orb_detector_;
  cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> ftracker_;
  std::vector<cv::cuda::GpuMat> detection_masks_;
  std::vector<cv::Mat> detection_masks_cpu_;
  bool feature_tracking_initialized_ = false;
  void DetectTrackFeatures();
  void DetectNewFeatures(const std::vector<int>& detection_mask_indices);
  std::vector<cv::KeyPoint> tracked_keypoints_cpu_;
  std::vector<cv::Scalar> tracked_feature_color_;
  cv::Mat tracked_descriptors_cpu_;
  cv::Mat KeypointsToMat(const std::vector<cv::KeyPoint>& keypoints);

  // Visualization
  cv::Mat DrawKeypoints(const cv::Mat& rgb_im); 
};

#endif

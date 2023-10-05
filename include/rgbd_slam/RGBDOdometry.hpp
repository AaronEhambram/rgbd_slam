#ifndef RGBDODOMETRY_H_
#define RGBDODOMETRY_H_
#define G2O_USE_VENDORED_CERES
#include <ros/ros.h>
#include "sensor_msgs/Image.h"
#include "cv_bridge/cv_bridge.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include "Eigen/Dense"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include "rgbd_slam/Frame.hpp"
#include <thread>

class RGBDOdometry
{
  public:
  RGBDOdometry(ros::NodeHandle& nh);
  void track(const sensor_msgs::ImageConstPtr& rgb_im_msg, const sensor_msgs::ImageConstPtr& depth_im_msg);
  std::vector<Frame*> frames_; 

  private:
  cv::Mat rgb_cam, rgb_distor, depth_cam, depth_distor, rgb_R_depth, rgb_t_depth;
  Eigen::Affine3d rgb_T_depth;
  void cv2eigen(cv::Mat& R, cv::Mat& t, Eigen::Affine3d& T);
  cv::cuda::GpuMat rgb_im_, rgb_im_prev_, depth_im_;
  void preprocess_images(const sensor_msgs::ImageConstPtr& rgb_im_ptr, const sensor_msgs::ImageConstPtr& depth_im_ptr);
  Frame* cur_frame_; 

  // feature detection and tracking:
  cv::Ptr<cv::cuda::ORB> orb_detector_;
  cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> ftracker_;
  cv::Ptr<cv::cuda::Filter> filter_;
  int horizontal_tiles_masks_ = 4; 
  int vertical_tiles_masks_ = 3;
  int rgb_im_width_, rgb_im_height_; 
  std::vector<cv::cuda::GpuMat> detection_masks_; 
  std::vector<cv::Mat> detection_masks_cpu_;
  void detect_track_features();
  bool feature_tracking_initialized = false; 
  void detect_further_features(); 
  void detect_features(cv::cuda::GpuMat& im, std::vector<cv::KeyPoint>& all_keypoints_cpu, cv::Mat& all_descriptors_cpu);
  int min_features_to_init_; 
  int min_features_in_mask_to_detect_new_; 
  void track_features(); 
  void landmarks_to_mat(cv::Mat& fs, std::vector<Landmark*>& landmarks, Frame* frame);
  double max_optical_flow_error_, min_feature_distribution_; 
  int min_features_in_mask_distribution_, min_features_to_track_only_;
  void delete_once_seen_landmarks(); 
  std::vector<Frame*> frames_with_once_seen_landmarks_;
  std::mutex mutex_frames_with_once_seen_landmarks_; 
  std::thread delete_once_seen_landmarks_thread_;
  int gauss_filter_size_; 

  // depth image processing
  cv::Mat transformed_depths_; 
  cv::Mat rgb_tpx_depth_; 
  void transform_depth_image();
  Eigen::Matrix3d K_rgb_, K_ir_,K_rgb_inv_, K_ir_inv_;
  void compute_3d_observations();

  // Dead-reckoning to run-time
  void compute_relative_pose_to_last_frame();
  Eigen::Affine3d origin_T_cur_pose_; 

  // SLAM-graph evaluation
  std::thread graph_evaluation_thread_;
  std::mutex frames_mutex_; 
  void graph_evaluation();
  int last_kf_idx_ = -1; 
  int last_processed_idx_ = -1; 

  // visualization
  void visualize_tracking(const sensor_msgs::ImageConstPtr& rgb_im_msg); 
  void publish_cur_landmarks();
  void publish_dead_reckoning_pose(); 
  std::shared_ptr<ros::Publisher> cur_landmark_publisher_;
  std::shared_ptr<ros::Publisher> dead_reckoning_publisher_;
};

#endif
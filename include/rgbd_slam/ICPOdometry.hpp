#ifndef ICPODOMETRY_H_
#define ICPODOMETRY_H_
#define G2O_USE_VENDORED_CERES
#include <ros/ros.h>
#include "sensor_msgs/Image.h"
#include "cv_bridge/cv_bridge.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include "Eigen/Dense"
#include "rgbd_slam/VoxelHashMap.hpp"
#include "sensor_msgs/PointCloud2.h"
#include <tf/transform_broadcaster.h>

class ICPOdometry
{
  public:
  ICPOdometry(ros::NodeHandle& nh);
  void track(const sensor_msgs::ImageConstPtr& rgb_im_msg, const sensor_msgs::ImageConstPtr& depth_im_msg);

  private:
  cv::Mat rgb_cam, rgb_distor, depth_cam, depth_distor, rgb_R_depth, rgb_t_depth;
  Eigen::Affine3d rgb_T_depth;
  Eigen::Matrix3d K_rgb_, K_ir_,K_rgb_inv_, K_ir_inv_;
  void cv2eigen(cv::Mat& R, cv::Mat& t, Eigen::Affine3d& T);
  void compute_pointcloud(const sensor_msgs::ImageConstPtr& depth_im_ptr);
  void prepare_next_time_step();
  bool map_initialized_ = false; 
  Eigen::Matrix<double,6,1> optimize_pose(const std::vector<Eigen::Vector3d>& local_points, const std::vector<std::pair<size_t,Voxel*>>& correspondences, Eigen::Affine3d& map_T_cur_predicted);
  Eigen::Matrix<double,6,1> optimize_pose_dist(const std::vector<Eigen::Vector3d>& local_points, const std::vector<std::pair<size_t,Voxel*>>& correspondences, Eigen::Affine3d& map_T_cur_predicted); 


  // current pointcloud
  std::vector<Eigen::Vector3d> points_;
  double cur_time_,prev_time_ = 0; 
  Eigen::Vector3d prev_trans_vel_, prev_eul_vel_; 
  Eigen::Affine3d map_T_cur_,map_T_prev_; 

  // the map
  double voxel_size_; 
  int max_points_per_voxel_; 
  VoxelHashMap map_;
  
  // rotation functions
  Eigen::Vector3d mat_to_eul_vect(const Eigen::Matrix3d& R);
  Eigen::Matrix3d eul_vect_to_mat(const Eigen::Vector3d& eul);
  inline double square(double x) { return x * x; }
  const double epsilon = 1e-8;
  Eigen::Matrix3d exp(const Eigen::Vector3d& omega);
  Eigen::Vector3d log(const Eigen::Matrix3d& R);
  Eigen::Matrix3d exp_times_vector_jacobian(const Eigen::Vector3d& omega, const Eigen::Matrix3d& R, const Eigen::Vector3d& local_point);
  Eigen::Matrix3d left_jacobian(const Eigen::Vector3d& omega);

  // Visualization
  void create_cloud(const std::vector<Eigen::Vector3d>& points, sensor_msgs::PointCloud2& cloud); 
  void show_points(const std::vector<Eigen::Vector3d>& points);
  void show_map_points();
  std::shared_ptr<ros::Publisher> points_publisher_;
  std::shared_ptr<ros::Publisher> map_publisher_;
  void show_correspondences(const std::vector<Eigen::Vector3d>& local_points, const Eigen::Affine3d map_T_cur_predicted, const std::vector<std::pair<size_t,Voxel*>>& correspondences);
  std::shared_ptr<ros::Publisher> correspondence_publisher_;
  tf::TransformBroadcaster tf_br_;
  void broadcast_tf();
};

#endif
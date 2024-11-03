#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.h"
#include <message_filters/synchronizer.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "cv_bridge/cv_bridge.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "Eigen/Dense"
#include <omp.h>
#include "rgbd_slam/types.hpp"
#include "rgbd_slam/RGBDSlam.hpp"

class RgbdSlamNode : public rclcpp::Node
{
public:
  RgbdSlamNode() : Node("rgbd_slam_node")
  {
    this->declare_parameter<std::string>("calibration_file", "");
    std::string calibration_file = this->get_parameter("calibration_file").as_string();
    this->declare_parameter<std::string>("rgb_topic", "");
    std::string rgb_topic = this->get_parameter("rgb_topic").as_string();
    this->declare_parameter<std::string>("depth_topic", "");
    std::string depth_topic = this->get_parameter("depth_topic").as_string();
    this->declare_parameter<int>("num_horizontal_tiles", 4);
    rgbd_slam_params_.num_horizontal_tiles = this->get_parameter("num_horizontal_tiles").as_int();
    this->declare_parameter<int>("num_vertical_tiles", 3);
    rgbd_slam_params_.num_vertical_tiles = this->get_parameter("num_vertical_tiles").as_int();
    this->declare_parameter<int>("im_width", 640);
    rgbd_slam_params_.im_width = this->get_parameter("im_width").as_int();
    this->declare_parameter<int>("im_height", 480);
    rgbd_slam_params_.im_height = this->get_parameter("im_height").as_int();
    this->declare_parameter<double>("max_optical_flow_error", 5.0);
    rgbd_slam_params_.max_optical_flow_error = this->get_parameter("max_optical_flow_error").as_double();
    this->declare_parameter<int>("min_features_in_mask_to_detect_new", 5);
    rgbd_slam_params_.min_features_in_mask_to_detect_new = this->get_parameter("min_features_in_mask_to_detect_new").as_int();

    // Read calibration data
    cv::Mat rgb_R_depth, rgb_t_depth;
    cv::FileStorage calibration_data(calibration_file, cv::FileStorage::READ);
    calibration_data["rgb_camera_matrix"] >> calibration_params_.rgb_cam;
    calibration_data["rgb_dist_coeff"] >> calibration_params_.rgb_distor;
    calibration_data["ir_camera_matrix"] >> calibration_params_.depth_cam;
    calibration_data["ir_dist_coeff"] >> calibration_params_.depth_distor;
    calibration_data["rgb_R_ir"] >> rgb_R_depth;
    calibration_data["rgb_t_ir"] >> rgb_t_depth;
    cv2eigen(rgb_R_depth, rgb_t_depth, calibration_params_.rgb_T_depth);
    calibration_data.release();

    // set SLAM object
    rgbd_slam_ptr_.reset(new RGBDSlam(calibration_params_, rgbd_slam_params_));

    // subscribers
    rgb_sub.subscribe(this, rgb_topic);
    depth_sub.subscribe(this, depth_topic);
    sync.reset(new Sync(MySyncPolicy(10), rgb_sub, depth_sub));
    sync->registerCallback(std::bind(&RgbdSlamNode::callback, this, std::placeholders::_1, std::placeholders::_2));
  }

private:
  // camera data from calibration
  CalibartionParameters calibration_params_{};

  // time synchronization
  message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub;
  message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub;
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> MySyncPolicy;
  typedef message_filters::Synchronizer<MySyncPolicy> Sync;
  std::shared_ptr<Sync> sync;

  // SLAM parameters
  RGBDSLAMParameters rgbd_slam_params_{};

  // SLAM object
  std::unique_ptr<RGBDSlam> rgbd_slam_ptr_;

  // functions
  void cv2eigen(cv::Mat &R, cv::Mat &t, Eigen::Affine3d &T)
  {
    T.matrix()(0, 0) = R.at<double>(0, 0);
    T.matrix()(1, 0) = R.at<double>(1, 0);
    T.matrix()(2, 0) = R.at<double>(2, 0);
    T.matrix()(0, 1) = R.at<double>(0, 1);
    T.matrix()(1, 1) = R.at<double>(1, 1);
    T.matrix()(2, 1) = R.at<double>(2, 1);
    T.matrix()(0, 2) = R.at<double>(0, 2);
    T.matrix()(1, 2) = R.at<double>(1, 2);
    T.matrix()(2, 2) = R.at<double>(2, 2);

    T.matrix()(0, 3) = t.at<double>(0);
    T.matrix()(1, 3) = t.at<double>(1);
    T.matrix()(2, 3) = t.at<double>(2);
  }

  void callback(const sensor_msgs::msg::Image::ConstSharedPtr &rgb_im_msg, const sensor_msgs::msg::Image::ConstSharedPtr &depth_im_msg)
  {
    // get the image through the cv:bridge
    cv_bridge::CvImageConstPtr rgb_cv_ptr, depth_cv_ptr;
    try
    {
      rgb_cv_ptr = cv_bridge::toCvShare(rgb_im_msg, "bgr8");
      depth_cv_ptr = cv_bridge::toCvShare(depth_im_msg);
    }
    catch (cv_bridge::Exception &e)
    {
      return;
    }

    cv::Mat rgb_im, depth_im;
    rgb_cv_ptr->image.copyTo(rgb_im);
    depth_cv_ptr->image.copyTo(depth_im);
    double timestamp = rgb_cv_ptr->header.stamp.sec + rgb_cv_ptr->header.stamp.nanosec * 1e-9;

    rgbd_slam_ptr_->Track(timestamp, rgb_im, depth_im);
  }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RgbdSlamNode>());
  rclcpp::shutdown();
  return 0;
}
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
//#include "rgbd_slam/RGBDOdometry.hpp"
//#include "rgbd_slam/ICPOdometry.hpp"

//std::unique_ptr<RGBDOdometry> rgbd_odom_ptr;
//std::unique_ptr<ICPOdometry> icp_odom_ptr; 

class RgbdSlamNode : public rclcpp::Node
{
  public: 
    RgbdSlamNode() : Node("rgbd_slam_node")
    {
      this->declare_parameter<std::string>("calibration_file","");
      std::string calibration_file = this->get_parameter("calibration_file").as_string();
      this->declare_parameter<std::string>("rgb_topic","");
      std::string rgb_topic = this->get_parameter("rgb_topic").as_string();
      this->declare_parameter<std::string>("depth_topic","");
      std::string depth_topic = this->get_parameter("depth_topic").as_string();

      // Read calibration data
      cv::FileStorage calibration_data(calibration_file, cv::FileStorage::READ);
      calibration_data["rgb_camera_matrix"] >> rgb_cam;
      calibration_data["rgb_dist_coeff"] >> rgb_distor;
      calibration_data["ir_camera_matrix"] >> depth_cam;
      calibration_data["ir_dist_coeff"] >> depth_distor;
      calibration_data["rgb_R_ir"] >> rgb_R_depth;
      calibration_data["rgb_t_ir"] >> rgb_t_depth;
      cv2eigen(rgb_R_depth,rgb_t_depth,rgb_T_depth);
      calibration_data.release();
      std::cout << "rgb_T_depth: " << std::endl << rgb_T_depth.matrix() << std::endl;

      // subscribers
      rgb_sub.subscribe(this, rgb_topic);
      depth_sub.subscribe(this, depth_topic);
      sync.reset(new Sync(MySyncPolicy(10),rgb_sub,depth_sub));
      sync->registerCallback(std::bind(&RgbdSlamNode::callback,this, std::placeholders::_1, std::placeholders::_2));
    }

    private:
    // camera data from calibration
    cv::Mat rgb_cam, rgb_distor, depth_cam, depth_distor, rgb_R_depth, rgb_t_depth;
    Eigen::Affine3d rgb_T_depth;

    // time synchronization
    message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> MySyncPolicy;
    typedef message_filters::Synchronizer<MySyncPolicy> Sync;
    std::shared_ptr<Sync> sync;

    // functions
    void cv2eigen(cv::Mat& R, cv::Mat& t, Eigen::Affine3d& T)
    {
      T.matrix()(0,0) = R.at<double>(0,0); 
      T.matrix()(1,0) = R.at<double>(1,0); 
      T.matrix()(2,0) = R.at<double>(2,0);
      T.matrix()(0,1) = R.at<double>(0,1); 
      T.matrix()(1,1) = R.at<double>(1,1); 
      T.matrix()(2,1) = R.at<double>(2,1);
      T.matrix()(0,2) = R.at<double>(0,2); 
      T.matrix()(1,2) = R.at<double>(1,2); 
      T.matrix()(2,2) = R.at<double>(2,2);

      T.matrix()(0,3) = t.at<double>(0); 
      T.matrix()(1,3) = t.at<double>(1); 
      T.matrix()(2,3) = t.at<double>(2);
    }

    void callback(const sensor_msgs::msg::Image::ConstSharedPtr& rgb_im_msg, const sensor_msgs::msg::Image::ConstSharedPtr& depth_im_msg)
    {
      // get the image through the cv:bridge
      cv_bridge::CvImageConstPtr rgb_cv_ptr, depth_cv_ptr;
      try
      {
        rgb_cv_ptr = cv_bridge::toCvShare(rgb_im_msg,"bgr8");
        depth_cv_ptr = cv_bridge::toCvShare(depth_im_msg);
      }
      catch(cv_bridge::Exception& e)
      {
        return;     
      }

      cv::Mat rgb_im, depth_im;
      rgb_cv_ptr->image.copyTo(rgb_im);
      depth_cv_ptr->image.copyTo(depth_im);

      // show the image
      cv::imshow("rgb image", rgb_im);
      cv::imshow("depth image", depth_im);
      cv::waitKey(1);
    }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RgbdSlamNode>());
  rclcpp::shutdown();
  return 0;
}
#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include <message_filters/synchronizer.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "cv_bridge/cv_bridge.h"
#include <opencv2/core.hpp>
#include "Eigen/Dense"
#include <omp.h>
#include "rgbd_slam/RGBDOdometry.hpp"
#include "rgbd_slam/ICPOdometry.hpp"

std::unique_ptr<RGBDOdometry> rgbd_odom_ptr;
std::unique_ptr<ICPOdometry> icp_odom_ptr; 

void callback(const sensor_msgs::ImageConstPtr& rgb_im_msg, const sensor_msgs::ImageConstPtr& depth_im_msg)
{
  // Call odometry object
  //rgbd_odom_ptr->track(rgb_im_msg,depth_im_msg);
  icp_odom_ptr->track(rgb_im_msg,depth_im_msg);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "rgbd_slam_node");
  ros::NodeHandle nh;

  //rgbd_odom_ptr.reset(new RGBDOdometry(nh)); 
  icp_odom_ptr.reset(new ICPOdometry(nh));

  std::string rgb_topic, depth_topic;  
  nh.getParam("rgbd_slam_node/rgb_topic", rgb_topic);
  nh.getParam("rgbd_slam_node/depth_topic", depth_topic);

  message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, rgb_topic, 1);
  message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, depth_topic, 1);
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
  // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
  message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), rgb_sub, depth_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2));

  ros::spin();
  return 0; 
}
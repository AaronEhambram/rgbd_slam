#include "rgbd_slam/ICPOdometry.hpp"
#include "sensor_msgs/point_cloud2_iterator.h"
#include "visualization_msgs/Marker.h"
#include <cmath>

ICPOdometry::ICPOdometry(ros::NodeHandle& nh)
{
  std::string calibration_file;  
  nh.getParam("rgbd_slam_node/calibration_file", calibration_file);
  nh.getParam("rgbd_slam_node/voxel_size", voxel_size_);
  nh.getParam("rgbd_slam_node/max_points_per_voxel", max_points_per_voxel_);

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

  // build the camera projection matrices
  K_ir_ <<  depth_cam.at<double>(0,0), depth_cam.at<double>(0,1), depth_cam.at<double>(0,2),
            depth_cam.at<double>(1,0), depth_cam.at<double>(1,1), depth_cam.at<double>(1,2),
            depth_cam.at<double>(2,0), depth_cam.at<double>(2,1), depth_cam.at<double>(2,2);
  K_ir_inv_ = K_ir_.inverse();
  std::cout << "K_ir_: " << std::endl << K_ir_ << std::endl; 
  K_rgb_ <<   rgb_cam.at<double>(0,0), rgb_cam.at<double>(0,1), rgb_cam.at<double>(0,2),
              rgb_cam.at<double>(1,0), rgb_cam.at<double>(1,1), rgb_cam.at<double>(1,2),
              rgb_cam.at<double>(2,0), rgb_cam.at<double>(2,1), rgb_cam.at<double>(2,2);
  K_rgb_inv_ = K_rgb_.inverse();
  std::cout << "K_rgb_: " << std::endl << K_rgb_ << std::endl; 

  // initilaze map
  map_ = VoxelHashMap(voxel_size_,max_points_per_voxel_);

  // initialize previous parameters
  prev_trans_vel_.setZero();
  prev_eul_vel_.setZero();
  map_T_prev_.setIdentity();
  map_T_cur_.setIdentity();

  // initialize publishers
  points_publisher_.reset(new ros::Publisher(nh.advertise<sensor_msgs::PointCloud2>("points", 1))); 
  map_publisher_.reset(new ros::Publisher(nh.advertise<sensor_msgs::PointCloud2>("map", 1)));
  correspondence_publisher_.reset(new ros::Publisher(nh.advertise<visualization_msgs::Marker>("correspondences", 1))); 
}

void ICPOdometry::cv2eigen(cv::Mat& R, cv::Mat& t, Eigen::Affine3d& T)
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

void ICPOdometry::track(const sensor_msgs::ImageConstPtr& rgb_im_msg, const sensor_msgs::ImageConstPtr& depth_im_msg)
{
  // generate the point cloud from depth image
  compute_pointcloud(depth_im_msg);

  VoxelHashMap local_points(voxel_size_,max_points_per_voxel_);
  std::vector<std::pair<size_t,Voxel*>> correspondences;
  if(map_initialized_)
  {
    // exists -> need to align to this map
    // compute VoxelHashMap for local cloud
    local_points.add_points(points_);
    // get voxelized local points
    std::vector<Eigen::Vector3d> voxelized_local_points = local_points.get_voxelized_points();

    // predict the pose from previous velocity (constant velocity model)
    Eigen::Affine3d prev_T_cur;
    prev_T_cur.translation() = prev_trans_vel_*(cur_time_-prev_time_);
    Eigen::Vector3d eul_pred = prev_eul_vel_*(cur_time_-prev_time_);
    prev_T_cur.linear() = eul_vect_to_mat(eul_pred);
    map_T_cur_ = map_T_prev_*prev_T_cur;

    Eigen::Matrix<double,6,1> update; update.setOnes();
    size_t iters = 0; 
    while(update.norm() > 0.001 && iters < 10)
    {
      // get correspondences
      correspondences.clear();
      map_.get_correspondences(map_T_cur_,voxelized_local_points,correspondences);

      // optimize pose
      update = optimize_pose(voxelized_local_points,correspondences,map_T_cur_);
      //update = optimize_pose_dist(voxelized_local_points,correspondences,map_T_cur_);

      iters++;
    }

    // visualize
    show_map_points();
    std::vector<Eigen::Vector3d> voxelized_points = local_points.get_voxelized_points();
    show_points(voxelized_points);
    show_correspondences(voxelized_points, map_T_cur_,correspondences);
    broadcast_tf();

    // Add new points to map
    if(update.norm() < 0.001)
    {
      map_.add_transformed_points(map_T_cur_,points_);
    }
  }
  else
  {
    // No local map -> first pose
    map_.add_points(points_);
    map_initialized_ = true; 
  }

  prepare_next_time_step();
}

void ICPOdometry::compute_pointcloud(const sensor_msgs::ImageConstPtr& depth_im_ptr)
{
  cur_time_ = depth_im_ptr->header.stamp.toSec();
  cv_bridge::CvImageConstPtr depth_cv_ptr;
  try
  {
    depth_cv_ptr = cv_bridge::toCvShare(depth_im_ptr);
  }
  catch(cv_bridge::Exception& e)
  {
    return;     
  }
  // images are links, NO COPY!
  const cv::Mat& depth_im = depth_cv_ptr->image;

  // Basically we are just interested in the depth image
  double& fx_depth = depth_cam.at<double>(0,0);
  double& fy_depth = depth_cam.at<double>(1,1);
  double& cx_depth = depth_cam.at<double>(0,2);
  double& cy_depth = depth_cam.at<double>(1,2);
  std::vector<Eigen::Vector3d> points;
  points.resize(depth_im.rows*depth_im.cols);
  #pragma omp parallel for collapse(2)
  for (int y=0; y<depth_im.rows; y++)
  {
    for (int x=0; x<depth_im.cols; x++)
    {
      Eigen::Vector3d& p = points[y*depth_im.cols+x];
      //get the depth
      unsigned short depth = depth_im.at<unsigned short>(y, x);
      p << (double)depth*((double)x-cx_depth)/fx_depth, (double)depth*((double)y-cy_depth)/fy_depth, (double)depth;
      p = p*0.001; // mm to m!
    }
  }
  // only compy points that are not (0,0,0)
  points_.clear(); 
  points_.reserve(points.size());
  for(const Eigen::Vector3d& p : points)
  {
    if(p.norm() > 0)
    {
      points_.emplace_back(p);
    }
  }
}

void ICPOdometry::prepare_next_time_step()
{
  if(prev_time_ > 0)
  {
    Eigen::Affine3d prev_T_cur = map_T_prev_.inverse()*map_T_cur_;
    // compute translation velocity
    prev_trans_vel_ = (prev_T_cur.translation())/(cur_time_-prev_time_);
    // compute rotation velocity
    prev_eul_vel_ = mat_to_eul_vect(prev_T_cur.linear())/(cur_time_-prev_time_);
  }
  prev_time_ = cur_time_; 
  map_T_prev_ = map_T_cur_; 
}

Eigen::Matrix<double,6,1> ICPOdometry::optimize_pose_dist(const std::vector<Eigen::Vector3d>& local_points, const std::vector<std::pair<size_t,Voxel*>>& correspondences, Eigen::Affine3d& map_T_cur_predicted)
{
  std::vector<double> residuals(correspondences.size());
  std::vector<Eigen::Matrix<double,1,6>> jacobians(correspondences.size());
  std::vector<double> weights(correspondences.size());
  Eigen::Matrix<double,6,1> pose_params;
  pose_params.template head<3>() = map_T_cur_predicted.translation();
  pose_params.template tail<3>() = log(map_T_cur_predicted.linear());
  double gm_th = 0.05; 
  #pragma omp parallel for
  for(int i = 0; i < correspondences.size(); i++)
  {
    // get corresponding points
    Eigen::Vector3d m_p_source = map_T_cur_predicted*local_points[correspondences[i].first];
    Eigen::Vector3d m_p_target = correspondences[i].second->points_[0];
    Eigen::Vector3d err = m_p_source-m_p_target;
    // residual
    residuals[i] = err.squaredNorm(); 
    // jacobian
    Eigen::Matrix<double,1,6>& J_r = jacobians[i];
    
    // translation
    J_r.setZero();
    J_r.block<1, 3>(0,0) = 2.0 * err.transpose() * Eigen::Matrix3d::Identity();

    // rotation
    Eigen::Matrix3d derr_dw;
    derr_dw = exp_times_vector_jacobian(pose_params.template tail<3>(), map_T_cur_predicted.linear(), local_points[correspondences[i].first]);
    J_r.block<1,3>(0,3) = 2.0 * err.transpose() * derr_dw;

    weights[i] = square(gm_th)/square(gm_th + residuals[i]);
  }
  // compute jacobian, residual sum
  Eigen::Matrix<double,6,6> JTJ; JTJ.setZero();
  Eigen::Matrix<double,6,1> JTr; JTr.setZero();
  for(int i = 0; i < correspondences.size(); i++)
  {
    double& r = residuals[i];
    Eigen::Matrix<double,1,6>& J = jacobians[i]; 
    double& w = weights[i]; 
    JTJ += J.transpose() * w * J; 
    JTr += J.transpose() * w * r; 
  }
  // solve equation system
  Eigen::Matrix<double,6,1> dx = JTJ.ldlt().solve(-JTr);
  pose_params = pose_params + dx; 
  map_T_cur_predicted.translation() = pose_params.template head<3>();
  map_T_cur_predicted.linear() = exp(pose_params.template tail<3>());
  return dx; 
}

Eigen::Matrix<double,6,1> ICPOdometry::optimize_pose(const std::vector<Eigen::Vector3d>& local_points, const std::vector<std::pair<size_t,Voxel*>>& correspondences, Eigen::Affine3d& map_T_cur_predicted)
{
  std::vector<Eigen::Vector3d> residuals(correspondences.size());
  std::vector<Eigen::Matrix<double,3,6>> jacobians(correspondences.size());
  std::vector<double> weights(correspondences.size());
  Eigen::Matrix<double,6,1> pose_params;
  pose_params.template head<3>() = map_T_cur_predicted.translation();
  pose_params.template tail<3>() = log(map_T_cur_predicted.linear());
  double gm_th = 0.05; 
  #pragma omp parallel for
  for(int i = 0; i < correspondences.size(); i++)
  {
    // get corresponding points
    Eigen::Vector3d m_p_source = map_T_cur_predicted*local_points[correspondences[i].first];
    Eigen::Vector3d m_p_target = correspondences[i].second->points_[0];
    // residual
    residuals[i] = m_p_source-m_p_target; 
    // jacobian
    Eigen::Matrix<double,3,6>& J_r = jacobians[i];
    J_r.setZero();
    J_r.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity(); // translation part
    //J_r.block<3,3>(0,3) <<  0, local_points[correspondences[i].first].z(), -local_points[correspondences[i].first].y(),
    //                        -local_points[correspondences[i].first].z(), 0, local_points[correspondences[i].first].x(),
    //                        local_points[correspondences[i].first].y(), -local_points[correspondences[i].first].x(), 0; //**************CKECK THIS!!! WHY???!
    J_r.block<3,3>(0,3) = exp_times_vector_jacobian(pose_params.template tail<3>(), map_T_cur_predicted.linear(), local_points[correspondences[i].first]);

    weights[i] = square(gm_th)/square(gm_th + residuals[i].squaredNorm());
  }
  // compute jacobian, residual sum
  Eigen::Matrix<double,6,6> JTJ; JTJ.setZero();
  Eigen::Matrix<double,6,1> JTr; JTr.setZero();
  for(int i = 0; i < correspondences.size(); i++)
  {
    Eigen::Vector3d& r = residuals[i];
    Eigen::Matrix<double,3,6>& J = jacobians[i]; 
    double& w = weights[i]; 
    JTJ += J.transpose() * w * J; 
    JTr += J.transpose() * w * r; 
  }
  // solve equation system
  Eigen::Matrix<double,6,1> dx = JTJ.ldlt().solve(-JTr);
  pose_params = pose_params + dx; 
  map_T_cur_predicted.translation() = pose_params.template head<3>();
  map_T_cur_predicted.linear() = exp(pose_params.template tail<3>());
  return dx; 
}

/******Rotation functions*******/

Eigen::Vector3d ICPOdometry::mat_to_eul_vect(const Eigen::Matrix3d& R)
{
  Eigen::Vector3d eul; 
  eul(0) = atan2(R(2,1),R(2,2));
  eul(1) = -asin(R(2,0));
  eul(2) = atan2(R(1,0),R(0,0));
  return eul; 
}

Eigen::Matrix3d ICPOdometry::eul_vect_to_mat(const Eigen::Vector3d& eul)
{
  Eigen::Quaternion<double> q = Eigen::AngleAxisd(eul(2),Eigen::Vector3d::UnitZ())
                            *Eigen::AngleAxisd(eul(1),Eigen::Vector3d::UnitY())
                            *Eigen::AngleAxisd(eul(0),Eigen::Vector3d::UnitX());
  return q.matrix(); 
}

Eigen::Matrix3d ICPOdometry::exp(const Eigen::Vector3d& omega)
{
  double theta = omega.norm(); 
  Eigen::Quaterniond q;
  if(abs(theta) < epsilon)
  {
    double theta_sq = theta * theta;
    double theta_po4 = theta_sq * theta_sq;
    double imag = double(0.5) - double(1.0 / 48.0) * theta_sq + double(1.0 / 3840.0) * theta_po4;
    double real = double(1) - double(1.0 / 8.0) * theta_sq + double(1.0 / 384.0) * theta_po4;
    q = Eigen::Quaterniond(real,omega.x()*imag,omega.y()*imag,omega.z()*imag);
  }
  else
  {
    double half_theta = double(0.5)*theta;
    q = Eigen::Quaterniond(cos(half_theta),omega.x()/theta*sin(half_theta),omega.y()/theta*sin(half_theta),omega.z()/theta*sin(half_theta));
  }
  return q.matrix();
}

Eigen::Vector3d ICPOdometry::log(const Eigen::Matrix3d& R)
{
  double R_trace = R.trace();
  Eigen::Vector3d omega(R(2,1)-R(1,2),R(0,2)-R(2,0),R(1,0)-R(0,1)); 
  double theta = acos((R_trace-double(1.0))/double(2.0));
  if(3.0-R_trace < epsilon)
  {
    omega = double(0.5)*(double(1.0) + double(1.0/6.0)*square(theta) + double(7.0)/double(360.0)*square(square(theta)))*omega; 
  }
  else
  {
    omega = theta/(double(2.0)*sin(theta))*omega; 
  }
  return omega; 
}

Eigen::Matrix3d ICPOdometry::left_jacobian(const Eigen::Vector3d& omega)
{
  double theta = omega.norm(); 
  if(abs(theta) < epsilon)
  {
    return (Eigen::Matrix3d::Identity());
  }
  else
  {
    double theta_sq = theta*theta; 
    double theta_3 = theta_sq*theta; 
    Eigen::Matrix3d Omega; 
    Omega <<  0, -omega.z(), omega.y(),
              omega.z(), 0, -omega.x(),
              -omega.y(), omega.x(), 0; 
    return (Eigen::Matrix3d::Identity() + (1.0-cos(theta))/theta_sq * Omega + (theta-sin(theta))/theta_3 * Omega * Omega);
  } 
}

Eigen::Matrix3d ICPOdometry::exp_times_vector_jacobian(const Eigen::Vector3d& omega, const Eigen::Matrix3d& R, const Eigen::Vector3d& local_point)
{
  Eigen::Matrix3d J_l = left_jacobian(omega);
  Eigen::Vector3d s = R*local_point;
  Eigen::Matrix3d Rs_skew; 
  Rs_skew <<  0, -s.z(), s.y(),
              s.z(), 0, -s.x(),
              -s.y(), s.x(), 0;
  return (-Rs_skew)*J_l;
}

/*******Visualization********/
void ICPOdometry::create_cloud(const std::vector<Eigen::Vector3d>& points, sensor_msgs::PointCloud2& cloud)
{
  // register color to pointcloud: https://answers.ros.org/question/219876/using-sensor_msgspointcloud2-natively/
  //declare message and sizes
  cloud.header.frame_id = "depth_frame";
  cloud.header.stamp = ros::Time::now();
  cloud.width  = points.size();
  cloud.height = 1;
  cloud.is_bigendian = false;
  cloud.is_dense = false; // there may be invalid points
  //for fields setup
  sensor_msgs::PointCloud2Modifier modifier(cloud);
  modifier.setPointCloud2FieldsByString(1,"xyz");
  modifier.resize(points.size());
  //iterators
  sensor_msgs::PointCloud2Iterator<float> out_x(cloud, "x");
  sensor_msgs::PointCloud2Iterator<float> out_y(cloud, "y");
  sensor_msgs::PointCloud2Iterator<float> out_z(cloud, "z");
  // store to cloud
  for (int i=0; i<points.size(); i++)
  {
    //store xyz in point cloud
    *out_x = (float)points[i](0);
    *out_y = (float)points[i](1);
    *out_z = (float)points[i](2);

    //increment
    ++out_x;
    ++out_y;
    ++out_z;
  } 
}

void ICPOdometry::show_points(const std::vector<Eigen::Vector3d>& points)
{
  sensor_msgs::PointCloud2 cloud;
  create_cloud(points,cloud);
  cloud.header.frame_id = "depth_frame";
  points_publisher_->publish(cloud);
}

void ICPOdometry::show_map_points()
{
  std::vector<Eigen::Vector3d> voxelized_points = map_.get_voxelized_points();
  sensor_msgs::PointCloud2 cloud;
  create_cloud(voxelized_points,cloud);
  cloud.header.frame_id = "map";
  map_publisher_->publish(cloud);
}

void ICPOdometry::show_correspondences(const std::vector<Eigen::Vector3d>& local_points, const Eigen::Affine3d map_T_cur_predicted, const std::vector<std::pair<size_t,Voxel*>>& correspondences)
{
  visualization_msgs::Marker marker;
  marker.header.frame_id = "map";
  marker.header.stamp = ros::Time();
  marker.ns = "correspondences";
  marker.id = 0;
  marker.type = visualization_msgs::Marker::LINE_LIST;
  marker.pose.position.x = 0;
  marker.pose.position.y = 0;
  marker.pose.position.z = 0;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 0.001;
  marker.scale.y = 0.001;
  marker.scale.z = 0.001;
  marker.color.a = 1.0; // Don't forget to set the alpha!
  marker.color.r = 0.0;
  marker.color.g = 1.0;
  marker.color.b = 0.0;
  marker.points.reserve(correspondences.size()*2);
  for(const std::pair<size_t,Voxel*>& pair : correspondences)
  {
    const Eigen::Vector3d& cur_point = local_points[pair.first];
    Eigen::Vector3d map_transformed_point = map_T_cur_predicted*cur_point;
    const Eigen::Vector3d& map_point = pair.second->points_[0];
    geometry_msgs::Point p1;
    p1.x = map_point(0); p1.y = map_point(1); p1.z = map_point(2);
    geometry_msgs::Point p2;
    p2.x = map_transformed_point(0); p2.y = map_transformed_point(1); p2.z = map_transformed_point(2);
    marker.points.emplace_back(p1);
    marker.points.emplace_back(p2);
  }
  correspondence_publisher_->publish(marker);
}

void ICPOdometry::broadcast_tf()
{
  tf::Transform transform;
  transform.setOrigin(tf::Vector3(map_T_cur_.translation().x(),map_T_cur_.translation().y(),map_T_cur_.translation().z()));
  Eigen::Quaterniond q(map_T_cur_.linear());
  tf::Quaternion q_tf(q.x(),q.y(),q.z(),q.w());
  transform.setRotation(q_tf);
  tf_br_.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map", "depth_frame"));
}


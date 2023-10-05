#include "rgbd_slam/RGBDOdometry.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/cudawarping.hpp>
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/point_cloud2_iterator.h"
#include <g2o/core/block_solver.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include "g2o/core/sparse_optimizer_terminate_action.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include <g2o/core/optimization_algorithm_levenberg.h>
#include "geometry_msgs/PoseStamped.h"

RGBDOdometry::RGBDOdometry(ros::NodeHandle& nh)
{
  std::string calibration_file;  
  nh.getParam("rgbd_slam_node/calibration_file", calibration_file);
  nh.getParam("rgbd_slam_node/horizontal_tiles_masks", horizontal_tiles_masks_);
  nh.getParam("rgbd_slam_node/vertical_tiles_masks", vertical_tiles_masks_);
  nh.getParam("rgbd_slam_node/rgb_im_width", rgb_im_width_);
  nh.getParam("rgbd_slam_node/rgb_im_height", rgb_im_height_);
  nh.getParam("rgbd_slam_node/min_features_to_init", min_features_to_init_);
  nh.getParam("rgbd_slam_node/min_features_in_mask_to_detect_new", min_features_in_mask_to_detect_new_);
  nh.getParam("rgbd_slam_node/max_optical_flow_error", max_optical_flow_error_);
  nh.getParam("rgbd_slam_node/min_features_in_mask_distribution", min_features_in_mask_distribution_);
  nh.getParam("rgbd_slam_node/min_feature_distribution", min_feature_distribution_);
  nh.getParam("rgbd_slam_node/min_features_to_track_only", min_features_to_track_only_);
  nh.getParam("rgbd_slam_node/gauss_filter_size", gauss_filter_size_);

  // Save the node-handle to publish messages
  cur_landmark_publisher_.reset(new ros::Publisher(nh.advertise<sensor_msgs::PointCloud2>("cur_landmarks", 1))); 
  dead_reckoning_publisher_.reset(new ros::Publisher(nh.advertise<geometry_msgs::PoseStamped>("dead_reckoning_pose", 1)));

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

  // build the translation matrix to shift the depth image
  const double& dx = rgb_T_depth.translation()(0);
  const double& dy = rgb_T_depth.translation()(1);
  rgb_tpx_depth_ = (cv::Mat_<double>(2,3) << 1, 0, dx*rgb_cam.at<double>(0,0), 0, 1, dy*rgb_cam.at<double>(1,1));

  // Get GPU information
  int gpucount = cv::cuda::getCudaEnabledDeviceCount();
  if(gpucount != 0){
      std::cout << "no. of gpu = " << gpucount << std::endl;
  }
  else
  {
      std::cout << "There is no CUDA supported GPU" << std::endl;
      return;
  }
  cv::cuda::setDevice(0);
  //cuda::resetDevice();
  enum cv::cuda::FeatureSet arch_avail;
  if(cv::cuda::TargetArchs::builtWith(arch_avail))
      std::cout << "yes, this Gpu arch is supported" << std::endl;
  cv::cuda::DeviceInfo deviceinfo;
  std::cout << "GPU: "<< deviceinfo.cv::cuda::DeviceInfo::name() << std::endl;

  // create detection masks
  int horizontal_tiles = horizontal_tiles_masks_;
  int vertical_tiles = vertical_tiles_masks_;
  int mask_width = std::floor(rgb_im_width_/horizontal_tiles); 
  int mask_height = std::floor(rgb_im_height_/vertical_tiles);
  detection_masks_.resize(horizontal_tiles*vertical_tiles);
  detection_masks_cpu_.resize(horizontal_tiles*vertical_tiles);
  int mask_counter = 0; 
  cv::Size l_im_size(rgb_im_width_,rgb_im_height_);
  for(int h = 0; h < horizontal_tiles; ++h)
  {
    for(int v = 0; v < vertical_tiles; ++v)
    {
      cv::Mat& mask = detection_masks_cpu_[mask_counter];
      mask = cv::Mat::zeros(l_im_size, CV_8U); 
      cv::Mat roi(mask, cv::Rect(h*mask_width,v*mask_height,mask_width,mask_height));
      roi = cv::Scalar(255);
      detection_masks_[mask_counter].upload(mask);
      mask_counter++; 
    }
  }

  // initilialze dead-reckoning pose
  origin_T_cur_pose_.setIdentity();

  // create feature detector and tracker
  orb_detector_ = cv::cuda::ORB::create(80,1.2f,5,31,0,2,0,31,20,false);
  ftracker_ = cv::cuda::SparsePyrLKOpticalFlow::create();
  filter_ = cv::cuda::createGaussianFilter(CV_8UC1,CV_8UC1,cv::Size(gauss_filter_size_,gauss_filter_size_),((double)gauss_filter_size_-1.0/6.0));

  // start thread for deleting once seen landmarks for the frames in the vector frames_with_once_seen_landmarks
  delete_once_seen_landmarks_thread_ = std::thread(&RGBDOdometry::delete_once_seen_landmarks, this);

  // start graph evaluation thread
  graph_evaluation_thread_ = std::thread(&RGBDOdometry::graph_evaluation, this);
}

void RGBDOdometry::cv2eigen(cv::Mat& R, cv::Mat& t, Eigen::Affine3d& T)
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

void RGBDOdometry::transform_depth_image()
{
  cv::cuda::GpuMat transformed_depths(depth_im_.size(), CV_16UC1);
  transformed_depths.setTo(0); 
  cv::cuda::warpAffine(depth_im_,transformed_depths,rgb_tpx_depth_,depth_im_.size());
  transformed_depths.download(transformed_depths_);
}

void RGBDOdometry::preprocess_images(const sensor_msgs::ImageConstPtr& rgb_im_ptr, const sensor_msgs::ImageConstPtr& depth_im_ptr)
{
  cv_bridge::CvImageConstPtr rgb_cv_ptr, depth_cv_ptr;
  try
  {
    rgb_cv_ptr = cv_bridge::toCvShare(rgb_im_ptr,"bgr8");
    depth_cv_ptr = cv_bridge::toCvShare(depth_im_ptr);
  }
  catch(cv_bridge::Exception& e)
  {
    return;     
  }
  cv::cuda::GpuMat rgb_im_color; 
  rgb_im_color.upload(rgb_cv_ptr->image); // only rgb-image with GPU-processing
  depth_im_ .upload(depth_cv_ptr->image);
  transform_depth_image();
  cv::cuda::cvtColor(rgb_im_color, rgb_im_, CV_BGR2GRAY);
  filter_->apply(rgb_im_,rgb_im_);
}

void RGBDOdometry::detect_features(cv::cuda::GpuMat& im, std::vector<cv::KeyPoint>& all_keypoints_cpu, cv::Mat& all_descriptors_cpu)
{
  // check which of the masks have how many features
  std::vector<int> features_in_mask(detection_masks_cpu_.size(),0);
  for(Landmark* l : cur_frame_->seen_landmarks_)
  {
    cv::Point2f& f = l->obsv_map_[cur_frame_]->f_;
    for(int i = 0; i < detection_masks_cpu_.size(); ++i)
    {
      cv::Mat& mask = detection_masks_cpu_[i];
      int v = (int) mask.at<uchar>(f.y,f.x);
      if(v > 100) // pixel value is 255
      {
        features_in_mask[i]++; 
      }
    }
  }

  std::vector<int> mask_indices_for_detection; 
  for(int i = 0; i < detection_masks_cpu_.size(); ++i)
  { 
    if(features_in_mask[i] < min_features_in_mask_to_detect_new_)
    {
      mask_indices_for_detection.push_back(i);
    }
  }
  // detect features
  for(int i = 0; i < mask_indices_for_detection.size(); ++i)
  {
    int mask_counter = mask_indices_for_detection[i]; // only indices less than min_features_in_mask_ for detection
    cv::cuda::GpuMat keypoints; // -> CV_32FC1 [#feature x 6]
    cv::cuda::GpuMat descriptors; // -> CV_8UC1 [32 x #feature]
    orb_detector_->detectAndComputeAsync(im,detection_masks_[mask_counter],keypoints,descriptors);
    std::vector<cv::KeyPoint> keypoints_cpu;
    orb_detector_->convert(keypoints,keypoints_cpu);
    all_keypoints_cpu.insert(all_keypoints_cpu.end(), keypoints_cpu.begin(), keypoints_cpu.end());
    cv::Mat descriptors_cpu;
    descriptors.download(descriptors_cpu);
    all_descriptors_cpu.push_back(descriptors_cpu);
  }
}

void RGBDOdometry::detect_further_features()
{
  //detect new features in the image
  std::vector<cv::KeyPoint> keypoints_cpu;
  cv::Mat descriptors_cpu;
  detect_features(rgb_im_, keypoints_cpu, descriptors_cpu);

  // the new landmarks are added to the tracked_landmarks_!
  // Each stereo feature is a new landmark with an observation from the current frame
  std::vector<Landmark*> new_landmarks(keypoints_cpu.size());
  std::vector<Observation*> new_observations(keypoints_cpu.size());
  #pragma omp parallel for 
  for(int i = 0; i < keypoints_cpu.size(); i++)
  {
    Landmark* new_landmark = new Landmark();
    Observation* new_observation = new Observation();
    new_observation->f_ = keypoints_cpu[i].pt;
    new_observation->frame_ = cur_frame_;
    new_observation->landmark_ = new_landmark;
    new_landmark->obsv_map_[cur_frame_] = new_observation;
    new_landmark->color_ = cv::Scalar(rand() % 255 + 0, rand() % 255 + 0, rand() % 255 + 0);

    // save to intermediate vectors
    new_landmarks[i] = new_landmark;
    new_observations[i] = new_observation;
  }
  cur_frame_->is_keyframe_ = true;
  cur_frame_->seen_landmarks_.insert(cur_frame_->seen_landmarks_.end(), new_landmarks.begin(), new_landmarks.end());

  //std::cout << "detect: " << cur_frame_->seen_landmarks_.size() << std::endl;
}

void RGBDOdometry::landmarks_to_mat(cv::Mat& fs, std::vector<Landmark*>& landmarks, Frame* frame)
{
  fs = cv::Mat(cv::Size(landmarks.size(),1), CV_32FC2);
  cv::Vec2f* features_cpu_col = fs.ptr<cv::Vec2f>(0);
  #pragma omp parallel for
  for(int i = 0; i<landmarks.size(); i++)
  {
    Landmark* l = landmarks[i];
    Observation* obs = l->obsv_map_[frame];
    cv::Point2f& f = obs->f_;
    cv::Vec2f kp;
    kp(0) = f.x;
    kp(1) = f.y;
    features_cpu_col[i] = kp;
  }
}

void RGBDOdometry::track_features()
{
  // tracked landmarks are those landmarks that were seen in the last frame
  Frame* last_frame = frames_.back(); // the last entry of graph_frames_ is the last frame (previous iteration)
  std::vector<Landmark*>& tracked_landmarks = last_frame->seen_landmarks_; // tracked landmarks are those that were seen in the last frame!
  cv::Mat features_cpu;
  landmarks_to_mat(features_cpu,tracked_landmarks,last_frame); // -> use the gpuMat to replace features

  // upload to gpumats
  cv::cuda::GpuMat features; 
  features.upload(features_cpu);

  // first try to track as many features as possible
  cv::cuda::GpuMat features_new, out, err;
  ftracker_->calc(rgb_im_prev_,rgb_im_,features,features_new, out, err);

  // download tracking results to CPU
  cv::Mat out_cpu(out.size(), out.type()), features_new_cpu(features_new.size(), features_new.type()), err_cpu;
  out.download(out_cpu); 
  features_new.download(features_new_cpu);
  err.download(err_cpu); 

  // save the well tracked features
  const uchar* out_cpu_col = out_cpu.ptr<uchar>(0);
  const cv::Vec2f* features_new_cpu_col = features_new_cpu.ptr<cv::Vec2f>(0); 
  const float* err_cpu_col = err_cpu.ptr<float>(0);
  std::vector<Observation*> new_observations; // ONLY new observations possible!
  for(int i = 0; i < out_cpu.cols; ++i)
  {
    if((int)out_cpu_col[i] == 1 && err_cpu_col[i] <= max_optical_flow_error_)
    {
      const cv::Vec2f& f = features_new_cpu_col[i];
      
      Landmark* correctly_tracked_landmark = tracked_landmarks[i];
      Observation* new_observation = new Observation();
      new_observation->f_.x = f(0);
      new_observation->f_.y = f(1);
      new_observation->frame_ = cur_frame_;
      new_observation->landmark_ = correctly_tracked_landmark;
      correctly_tracked_landmark->obsv_map_[cur_frame_] = new_observation;
      cur_frame_->seen_landmarks_.push_back(correctly_tracked_landmark);
    } 
  }

  cur_frame_->is_keyframe_ = false;
}

void RGBDOdometry::detect_track_features()
{
  if(!feature_tracking_initialized)
  {
    // Detect the features!
    detect_further_features(); 
    if(cur_frame_->seen_landmarks_.size() >= min_features_to_init_)
    {
      feature_tracking_initialized = true;
    }
  }
  else
  {
    // try to track features
    track_features();
    // check which of the masks have how many features
    std::vector<int> features_in_mask(detection_masks_cpu_.size(),0);
    for(Landmark* l : cur_frame_->seen_landmarks_)
    {
      cv::Point2f f;
      f = l->obsv_map_[cur_frame_]->f_;
      for(int i = 0; i < detection_masks_cpu_.size(); ++i)
      {
        cv::Mat& mask = detection_masks_cpu_[i];
        int v = (int) mask.at<uchar>(f.y,f.x);
        if(v > 100) // pixel value is 255
        {
          features_in_mask[i]++; 
        }
      }
    }

    double distribution_factor = 0; ; 
    for(int i = 0; i < features_in_mask.size(); ++i)
    {
      if(features_in_mask[i] > min_features_in_mask_distribution_)
      {
        distribution_factor = distribution_factor+1; 
      }
    }
    distribution_factor = distribution_factor/((double) features_in_mask.size());

    if( distribution_factor <= min_feature_distribution_ || cur_frame_->seen_landmarks_.size() <= min_features_to_track_only_)
    { 
      detect_further_features();
    }
    else
    {
      // if the previous frame was a keyframe -> delete the seen landmarks that are only seen once by this keyframe!
      if(frames_.back()->is_keyframe_)
      {
        mutex_frames_with_once_seen_landmarks_.lock();
        frames_with_once_seen_landmarks_.push_back(frames_.back());
        mutex_frames_with_once_seen_landmarks_.unlock();
      }
    }
  }
}

void RGBDOdometry::compute_3d_observations()
{
  // get projection parameters
  const double& cx = rgb_cam.at<double>(0,2);
  const double& cy = rgb_cam.at<double>(1,2);
  const double& fx = rgb_cam.at<double>(0,0);
  const double& fy = rgb_cam.at<double>(1,1);

  // iterate through each seen landmark
  #pragma omp parallel for
  for(int i = 0; i < cur_frame_->seen_landmarks_.size(); i++)
  {
    Landmark* l = cur_frame_->seen_landmarks_[i];
    // get feature location
    Observation* obs = l->obsv_map_[cur_frame_];
    cv::Point2f& px_f = obs->f_;
    // get corresponding depth
    unsigned short& depth_u = transformed_depths_.at<unsigned short>(px_f.y, px_f.x);
    double depth = (double)depth_u*0.001; // mm to m!
    // compute the 3D point
    obs->lcam_point_ << depth*(px_f.x - cx)/fx, depth*(px_f.y - cy)/fy, depth;
  }
}

void RGBDOdometry::track(const sensor_msgs::ImageConstPtr& rgb_im_msg, const sensor_msgs::ImageConstPtr& depth_im_msg)
{
  // prepare new frame as current frame
  cur_frame_ = new Frame();
  cur_frame_->timestamp_ = rgb_im_msg->header.stamp.toSec();

  // preprocessing: upload to GpuMat and convert color image to gray, warp the depth image (simple translation since depth and rgb are parallel)
  preprocess_images(rgb_im_msg, depth_im_msg);

  // Perform detection and tracking of the features
  detect_track_features();

  // compute 3D points for the tracked features in cur_frame_
  compute_3d_observations();

  // compute relative pose to last frame, initialize optimization with constant movement model
  if(frames_.size() > 0)
  {
    compute_relative_pose_to_last_frame();
  }

  // save to frames!
  frames_mutex_.lock();
  frames_.push_back(cur_frame_);
  frames_mutex_.unlock(); 
  // save the old images for the next iteration
  rgb_im_.copyTo(rgb_im_prev_);

  visualize_tracking(rgb_im_msg); 
}

/********Compute the relative transformation of cur_frame_ to last frame**********/
void RGBDOdometry::compute_relative_pose_to_last_frame()
{
  Frame* last_frame = frames_.back();
  // get the landmarks that are seen by last_frame and cur_frame_
  std::vector<Landmark*> common_landmarks; 
  common_landmarks.reserve(cur_frame_->seen_landmarks_.size());
  std::mutex common_landmarks_mutex; 
  last_frame->mutex_.lock(); // lock the mutex of the last_frame since it might be processed somewhere else
  #pragma omp parallel for
  for(int i = 0; i < cur_frame_->seen_landmarks_.size(); i++)
  {
    // get the landmark seen from cur_frame_
    Landmark* l = cur_frame_->seen_landmarks_[i];

    // check if l was seen in last_frame
    std::vector<Landmark*>::iterator landmark_find_it = std::find(last_frame->seen_landmarks_.begin(), last_frame->seen_landmarks_.end(), l);
    if(landmark_find_it != last_frame->seen_landmarks_.end())
    {
      // found the landmark! -> l is commonly observed
      common_landmarks_mutex.lock(); 
      common_landmarks.emplace_back(l);
      common_landmarks_mutex.unlock(); 
    }
  }
  last_frame->mutex_.unlock(); // release the last frame

  // TODO: Optimization of current frame by fixing the landmarks to the coordinates of the last frame and minimizing the reprojection error
  // setup optimizer (optimization graph)
  g2o::SparseOptimizer optimizer;
  std::unique_ptr<g2o::BlockSolverX::LinearSolverType> linearSolver; // BlockSolver_6_3
  linearSolver = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolverX::PoseMatrixType>>();//LinearSolverCSparse//LinearSolverCholmod//LinearSolverDense//LinearSolverEigen
  g2o::OptimizationAlgorithmLevenberg* solver;
  solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolverX>(std::move(linearSolver)));
  optimizer.setAlgorithm(solver);

  // insert cur_frame_ pose
  int vertex_id = 0;
  // compute the predicted pose in the last_frame based on constant motion model
  if(frames_.size() >= 2)
  {
    Frame* bef_last_frame = frames_[frames_.size()-2];
    // translation
    Eigen::Vector3d t = bef_last_frame->cam_prev_T_cam_cur_.translation();
    t = t/(last_frame->timestamp_ - bef_last_frame->timestamp_) * (cur_frame_->timestamp_ - last_frame->timestamp_);
    cur_frame_->cam_prev_T_cam_cur_.translation() = t; 
    // rotation
    Eigen::Matrix3d R = last_frame->cam_prev_T_cam_cur_.linear();
    Eigen::Vector3d r; 
    r(0) = atan2(R(2,1),R(2,2));
    r(1) = -asin(R(2,0));
    r(2) = atan2(R(1,0),R(0,0));
    r = r/(last_frame->timestamp_ - bef_last_frame->timestamp_) * (cur_frame_->timestamp_ - last_frame->timestamp_);
    Eigen::Quaternion<double> q = Eigen::AngleAxisd(r(2),Eigen::Vector3d::UnitZ())
                            *Eigen::AngleAxisd(r(1),Eigen::Vector3d::UnitY())
                            *Eigen::AngleAxisd(r(0),Eigen::Vector3d::UnitX());
    cur_frame_->cam_prev_T_cam_cur_.linear() = q.matrix();
  } 
  else
  {
    cur_frame_->cam_prev_T_cam_cur_ = last_frame->cam_prev_T_cam_cur_; 
  }
  g2o::VertexSE3Expmap* v_f_cur = new g2o::VertexSE3Expmap();
  Eigen::Affine3d cur_frame_T_last_frame = cur_frame_->cam_prev_T_cam_cur_.inverse();
  g2o::SE3Quat cur_frame_T_last_frame_quat(cur_frame_T_last_frame.linear(),cur_frame_T_last_frame.translation());
  v_f_cur->setEstimate(cur_frame_T_last_frame_quat);
  v_f_cur->setId(vertex_id);
  vertex_id++;
  v_f_cur->setFixed(false);       
  optimizer.addVertex(v_f_cur);

  // insert landmarks
  std::vector<g2o::EdgeSE3ProjectXYZ*> edges_vector(common_landmarks.size()); 
  int c = 0; 
  for(Landmark* l : common_landmarks)
  {
    if(l->obsv_map_[last_frame]->lcam_point_.norm() > 0)
    {
      // create landmark
      g2o::VertexPointXYZ* v_l = new g2o::VertexPointXYZ();
      v_l->setId(vertex_id);
      vertex_id++;
      g2o::Vector3 pos(l->obsv_map_[last_frame]->lcam_point_);
      v_l->setEstimate(pos);
      v_l->setMarginalized(false);
      v_l->setFixed(true);
      optimizer.addVertex(v_l);

      // insert the observation edge
      g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
      e->vertices()[0] = v_l;
      e->vertices()[1] = v_f_cur;
      double px = (double)l->obsv_map_[cur_frame_]->f_.x;
      double py = (double)l->obsv_map_[cur_frame_]->f_.y;
      g2o::Vector2 pix_obs(px, py);
      e->setMeasurement(pix_obs);
      Eigen::Matrix2d info;
      info.setIdentity();
      e->setInformation(info);
      g2o::RobustKernelGemanMcClure* rk = new g2o::RobustKernelGemanMcClure;
      e->setRobustKernel(rk);
      rk->setDelta(7.0); 
      e->fx = rgb_cam.at<double>(0,0);
      e->fy = rgb_cam.at<double>(1,1);
      e->cx = rgb_cam.at<double>(0,2);
      e->cy = rgb_cam.at<double>(1,2);
      optimizer.addEdge(e);
      edges_vector[c] = e; 
      c++; 
    }
    else
    {
      // no depth data is there for the feature!
    }
  }

  // optimize with Geman McClure Cost Function
  optimizer.setVerbose(false);
  optimizer.initializeOptimization();
  optimizer.optimize(20);

  Eigen::Isometry3d lcam_prev_T_lcam_cur_quat = v_f_cur->estimate().inverse();
  std::cout << lcam_prev_T_lcam_cur_quat.matrix() << std::endl << std::endl; 
  origin_T_cur_pose_ = origin_T_cur_pose_*lcam_prev_T_lcam_cur_quat;
  publish_dead_reckoning_pose();

  // TODO: Optimization with Geman-McClure kernel, then with saturated kernel
}

/***********Visualization*********/
void RGBDOdometry::visualize_tracking(const sensor_msgs::ImageConstPtr& rgb_im_msg)
{
  cv_bridge::CvImageConstPtr rgb_cv_ptr, depth_cv_ptr;
  try
  {
    rgb_cv_ptr = cv_bridge::toCvShare(rgb_im_msg,"bgr8");
  }
  catch(cv_bridge::Exception& e)
  {
    return;     
  }
  cv::Mat im;
  rgb_cv_ptr->image.copyTo(im);
  for(int i = 0; i < cur_frame_->seen_landmarks_.size(); i++)
  {
    Landmark* l = cur_frame_->seen_landmarks_[i]; 
    Observation* obs = l->obsv_map_[cur_frame_];

    // draw detected keypoint
    int radiusCircle = 5;
    int thickness = 3;
    cv::Point2f kp;
    kp = obs->f_;
    cv::circle(im, kp, radiusCircle, l->color_, thickness);
  }
  cv::imshow("Image with features", im);
  //cv::imshow("transformed depths", transformed_depths_);

  // create landmark cloud for currently seen landmarks
  //publish_cur_landmarks();

  cv::waitKey(1);
}

void RGBDOdometry::publish_cur_landmarks()
{
  //declare message and sizes
  sensor_msgs::PointCloud2 cloud;
  cloud.header.frame_id = "camera_rgb_frame";
  cloud.header.stamp = ros::Time::now();
  cloud.width  = cur_frame_->seen_landmarks_.size();
  cloud.height = 1;
  cloud.is_bigendian = false;
  cloud.is_dense = false; // there may be invalid points
  //for fields setup
  sensor_msgs::PointCloud2Modifier modifier(cloud);
  modifier.setPointCloud2FieldsByString(2,"xyz","rgb");
  modifier.resize(cur_frame_->seen_landmarks_.size());
  //iterators
  sensor_msgs::PointCloud2Iterator<float> out_x(cloud, "x");
  sensor_msgs::PointCloud2Iterator<float> out_y(cloud, "y");
  sensor_msgs::PointCloud2Iterator<float> out_z(cloud, "z");
  sensor_msgs::PointCloud2Iterator<uint8_t> out_r(cloud, "r");
  sensor_msgs::PointCloud2Iterator<uint8_t> out_g(cloud, "g");
  sensor_msgs::PointCloud2Iterator<uint8_t> out_b(cloud, "b");
  // store to cloud
  for (int i=0; i<cur_frame_->seen_landmarks_.size(); i++)
  {
    //store xyz in point cloud, transforming from image coordinates, (Z Forward to X Forward)
    *out_x = (float)cur_frame_->seen_landmarks_[i]->obsv_map_[cur_frame_]->lcam_point_(0);
    *out_y = (float)cur_frame_->seen_landmarks_[i]->obsv_map_[cur_frame_]->lcam_point_(1);
    *out_z = (float)cur_frame_->seen_landmarks_[i]->obsv_map_[cur_frame_]->lcam_point_(2);

    // store colors
    *out_r = cur_frame_->seen_landmarks_[i]->color_(2);
    *out_g = cur_frame_->seen_landmarks_[i]->color_(1);
    *out_b = cur_frame_->seen_landmarks_[i]->color_(0);

    //increment
    ++out_x;
    ++out_y;
    ++out_z;
    ++out_r;
    ++out_g;
    ++out_b;
  }
  cur_landmark_publisher_->publish(cloud);
}

void RGBDOdometry::publish_dead_reckoning_pose()
{
  geometry_msgs::PoseStamped msg; 
  msg.header.stamp = ros::Time(cur_frame_->timestamp_);
  msg.header.frame_id = "origin"; 
  msg.pose.position.x = origin_T_cur_pose_.translation().x(); 
  msg.pose.position.y = origin_T_cur_pose_.translation().y(); 
  msg.pose.position.z = origin_T_cur_pose_.translation().z(); 
  Eigen::Quaterniond q(origin_T_cur_pose_.linear());
  msg.pose.orientation.x = q.x();
  msg.pose.orientation.y = q.y();
  msg.pose.orientation.z = q.z();
  msg.pose.orientation.w = q.w();
  dead_reckoning_publisher_->publish(msg);
}

/****delete only once seen landmarks in a separate thread*****/
void RGBDOdometry::delete_once_seen_landmarks()
{
  while(ros::ok())
  {
    mutex_frames_with_once_seen_landmarks_.lock();
    if(frames_with_once_seen_landmarks_.size() > 2) // always delay the deletion by one frame -> so that mutex does not interfere with tracking
    {
      // get the frame for which we need to delete the once seen landmarks
      Frame* frame = frames_with_once_seen_landmarks_[0];
      frames_with_once_seen_landmarks_.erase(frames_with_once_seen_landmarks_.begin());
      mutex_frames_with_once_seen_landmarks_.unlock(); 

      // iterate through the seen landmarks 
      frame->mutex_.lock(); 
      for(int i = 0; i<frame->seen_landmarks_.size(); i++)
      {
        Landmark* l = frame->seen_landmarks_[i];
        if(l->obsv_map_.size() <= 1)
        {
          // delete the landmark and the corresponding observations!
          // delete in the vectors
          Observation* obs = l->obsv_map_[frame];
          frame->seen_landmarks_.erase(frame->seen_landmarks_.begin()+i);
          i--;
          // delete the objects
          delete obs;
          delete l;
        }
      }
      frame->mutex_.unlock();
    }
    else
    {
      mutex_frames_with_once_seen_landmarks_.unlock();
    }
    ros::Duration(0.5).sleep();
  }
}

/***********Graph Evaluation: Back-End*********/ 
void RGBDOdometry::graph_evaluation()
{
  while(ros::ok())
  {
    // copy the vetcor, so that front-end can continue
    //frames_mutex_.lock();
    std::vector<Frame*> frames_copy = frames_; 
    //frames_mutex_.unlock();
    if((int)frames_copy.size()-1 > last_processed_idx_)
    {
      for(int i = last_processed_idx_+1; i < frames_copy.size(); ++i)
      {
        if(frames_copy[i]->is_keyframe_)
        {
          // frame is keyframe! Last keyframe is at last_kf_idx_
          //std::cout << "KF!" << std::endl; 
          last_kf_idx_ = i; 
        }
        last_processed_idx_ = i;
      } 
    }
    else
    {
      ros::Duration(0.1).sleep(); 
    }
  }
}
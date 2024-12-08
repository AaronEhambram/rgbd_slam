#include "rgbd_slam/RGBDSlam.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iomanip>
#include <omp.h>

RGBDSlam::RGBDSlam(const CalibartionParameters &calibration_params, const RGBDSLAMParameters &slam_params)
{
  calibration_params_ = calibration_params;
  slam_params_ = slam_params;

  std::cout << "rgb_T_depth:\t" << std::endl
            << calibration_params_.rgb_T_depth.matrix() << std::endl;
  std::cout << "Image size:\t" << slam_params_.im_height << " x " << slam_params_.im_width << std::endl;
  std::cout << "Tiles:\t" << slam_params_.num_horizontal_tiles << " x " << slam_params_.num_vertical_tiles << std::endl;

  // Get GPU information
  int gpucount = cv::cuda::getCudaEnabledDeviceCount();
  if (gpucount != 0)
  {
    std::cout << "no. of gpu = " << gpucount << std::endl;
  }
  else
  {
    std::cout << "There is no CUDA supported GPU" << std::endl;
    return;
  }
  cv::cuda::setDevice(0);
  enum cv::cuda::FeatureSet arch_avail{};
  if (cv::cuda::TargetArchs::builtWith(arch_avail))
    std::cout << "yes, this Gpu arch is supported" << std::endl;
  cv::cuda::DeviceInfo deviceinfo;
  std::cout << "GPU: " << deviceinfo.cv::cuda::DeviceInfo::name() << std::endl;

  // create detection masks
  const int mask_width = std::floor(slam_params_.im_width / slam_params_.num_horizontal_tiles);
  const int mask_height = std::floor(slam_params_.im_height / slam_params_.num_vertical_tiles);
  const int num_tiles = slam_params_.num_horizontal_tiles * slam_params_.num_vertical_tiles;
  detection_masks_.resize(num_tiles);
  detection_masks_cpu_.resize(num_tiles);
  int mask_counter = 0;
  cv::Size im_size(slam_params_.im_width, slam_params_.im_height);
  for (int h = 0; h < slam_params_.num_horizontal_tiles; ++h)
  {
    for (int v = 0; v < slam_params_.num_vertical_tiles; ++v)
    {
      cv::Mat &mask = detection_masks_cpu_[mask_counter];
      mask = cv::Mat::zeros(im_size, CV_8U);
      cv::Mat roi(mask, cv::Rect(h * mask_width, v * mask_height, mask_width, mask_height));
      roi = cv::Scalar(255);
      detection_masks_[mask_counter].upload(mask);
      mask_counter++;
    }
  }

  // clear all containers
  tracked_keypoints_cpu_.clear();

  orb_detector_ = cv::cuda::ORB::create(80, 1.2f, 5, 31, 0, 2, 0, 31, 20, true);
  ftracker_ = cv::cuda::SparsePyrLKOpticalFlow::create();
  feature_tracking_initialized_ = false;

  // pose estimation
  start_T_cur.setIdentity();
}

cv::Mat RGBDSlam::KeypointsToMat(const std::vector<cv::KeyPoint> &keypoints)
{
  cv::Mat keypoints_mat = cv::Mat(cv::Size(keypoints.size(), 1), CV_32FC2);
  cv::Vec2f *features_cpu_col = keypoints_mat.ptr<cv::Vec2f>(0);
#pragma omp parallel for
  for (int i = 0; i < keypoints.size(); i++)
  {
    const cv::Point2f &f = keypoints[i].pt;
    cv::Vec2f kp;
    kp(0) = f.x;
    kp(1) = f.y;
    features_cpu_col[i] = kp;
  }
  return keypoints_mat;
}

void RGBDSlam::DetectNewFeatures(const std::vector<int> &detection_mask_indices)
{
  for (const int &mask_index : detection_mask_indices)
  {
    cv::cuda::GpuMat keypoints;   // -> CV_32FC1 [#feature x 6]
    cv::cuda::GpuMat descriptors; // -> CV_8UC1 [32 x #feature]
    orb_detector_->detectAndComputeAsync(rgb_im_, detection_masks_[mask_index], keypoints, descriptors);
    std::vector<cv::KeyPoint> keypoints_cpu;
    orb_detector_->convert(keypoints, keypoints_cpu);
    tracked_keypoints_cpu_.insert(tracked_keypoints_cpu_.end(), keypoints_cpu.begin(), keypoints_cpu.end());
    cv::Mat descriptors_cpu;
    descriptors.download(descriptors_cpu);
    tracked_descriptors_cpu_.push_back(descriptors_cpu);
    for (int new_kps_counter = 0; new_kps_counter < keypoints_cpu.size(); new_kps_counter++)
    {
      tracked_feature_color_.push_back(cv::Scalar(rand() % 255 + 0, rand() % 255 + 0, rand() % 255 + 0));
    }
  }
}

TrackingResult RGBDSlam::DetectTrackFeatures()
{
  TrackingResult tracking_result;
  if (!feature_tracking_initialized_)
  {
    // Detect new features in all detection masks
    std::vector<int> all_detection_mask_indices;
    all_detection_mask_indices.resize(detection_masks_.size());
    for (int i = 0; i < detection_masks_.size(); ++i)
    {
      all_detection_mask_indices[i] = i;
    }
    // detect the features
    DetectNewFeatures(all_detection_mask_indices);
    feature_tracking_initialized_ = true;
  }
  else
  {
    // tracking is initilized -> try to use optical flow
    cv::Mat features_cpu_mat = KeypointsToMat(tracked_keypoints_cpu_);

    // upload to gpumats
    cv::cuda::GpuMat features;
    features.upload(features_cpu_mat);

    // first try to track as many features as possible
    cv::cuda::GpuMat features_new, tracking_status, err;
    ftracker_->calc(rgb_im_prev_, rgb_im_, features, features_new, tracking_status, err);

    // download tracking results to CPU
    cv::Mat tracking_status_cpu(tracking_status.size(), tracking_status.type()), features_new_cpu(features_new.size(), features_new.type()), err_cpu;
    tracking_status.download(tracking_status_cpu);
    features_new.download(features_new_cpu);
    err.download(err_cpu);

    // save the well tracked features
    const uchar *tracking_status_cpu_col = tracking_status_cpu.ptr<uchar>(0);
    const cv::Vec2f *features_new_cpu_col = features_new_cpu.ptr<cv::Vec2f>(0);
    const float *err_cpu_col = err_cpu.ptr<float>(0);

    // Containers for current and prevous key points
    tracking_result.current_keypoints.reserve(tracked_keypoints_cpu_.size());
    tracking_result.previous_keypoints.reserve(tracked_keypoints_cpu_.size());

    // Container for the tracking error
    tracking_result.tracking_error.reserve(tracked_keypoints_cpu_.size());

    // Container for the color
    std::vector<cv::Scalar> tracked_feature_color;
    tracked_feature_color.reserve(tracked_feature_color_.size());

    // Container for the descriptors
    cv::Mat tracked_descriptors;

    // iterate through each keypoint and store the data to the containers
    for (int i = 0; i < tracking_status_cpu.cols; ++i)
    {
      if ((int)tracking_status_cpu_col[i] == 1 && err_cpu_col[i] <= slam_params_.max_optical_flow_error)
      {
        // get the new feature position
        const cv::Vec2f &f = features_new_cpu_col[i];

        // update the keypoint position
        tracking_result.previous_keypoints.emplace_back(tracked_keypoints_cpu_[i]);
        tracked_keypoints_cpu_[i].pt = {f(0), f(1)};
        tracking_result.current_keypoints.emplace_back(tracked_keypoints_cpu_[i]);

        // store error
        tracking_result.tracking_error.emplace_back(err_cpu_col[i]);
        // store descriptor
        tracked_descriptors.push_back(tracked_descriptors_cpu_.row(i));
        // store color
        tracked_feature_color.emplace_back(tracked_feature_color_[i]);
      }
    }

    // delete the untracked features by overwriting
    tracked_keypoints_cpu_ = tracking_result.current_keypoints;
    tracked_feature_color_ = tracked_feature_color;
    tracked_descriptors_cpu_ = cv::Mat(tracked_descriptors.size(), tracked_descriptors.type());
    tracked_descriptors.copyTo(tracked_descriptors_cpu_);

    // Identify masks with too less keypoints
    std::vector<int> features_in_mask(detection_masks_cpu_.size(), 0);
    for (const cv::KeyPoint &kp : tracked_keypoints_cpu_)
    {
      for (int i = 0; i < detection_masks_cpu_.size(); ++i)
      {
        cv::Mat &mask = detection_masks_cpu_[i];
        int v = (int)mask.at<uchar>(kp.pt.y, kp.pt.x);
        if (v > 100) // pixel value is 255
        {
          features_in_mask[i]++;
        }
      }
    }
    std::vector<int> new_detection_mask_indices;
    for (int i = 0; i < features_in_mask.size(); i++)
    {
      if (features_in_mask[i] < slam_params_.min_features_in_mask_to_detect_new)
      {
        new_detection_mask_indices.emplace_back(i);
      }
    }

    // detect the new features in the underrepresented masks
    DetectNewFeatures(new_detection_mask_indices);
  }

  return tracking_result;
}

std::optional<Eigen::Vector3d> RGBDSlam::Get3DPoint(const cv::KeyPoint &keypoint) const
{
  uint16_t depth_in_mm = depth_im_prev_.at<uint16_t>(keypoint.pt.y, keypoint.pt.x);
  std::optional<Eigen::Vector3d> point{};
  if (depth_in_mm > 0)
  {
    const double depth = 1e-3 * depth_in_mm;
    const double &fx_rgb = calibration_params_.rgb_cam.at<double>(0, 0);
    const double &fy_rgb = calibration_params_.rgb_cam.at<double>(1, 1);
    const double &cx_rgb = calibration_params_.rgb_cam.at<double>(0, 2);
    const double &cy_rgb = calibration_params_.rgb_cam.at<double>(1, 2);
    return Eigen::Vector3d{depth * ((double)keypoint.pt.x - cx_rgb) / fx_rgb, (double)depth * ((double)keypoint.pt.y - cy_rgb) / fy_rgb, depth};
  }
  return point;
}

double RGBDSlam::DetermineWeight(const float &tracking_error) const
{
  return std::exp(-std::pow(static_cast<double>(tracking_error), 2.0) / (2.0 * std::pow(slam_params_.max_optical_flow_error / 4.0, 2.0)));
}

Eigen::Affine3d RGBDSlam::EstimateRelativePose(const TrackingResult &tracking_result) const
{
  Eigen::MatrixXd A(tracking_result.previous_keypoints.size() * 2, 12);
  A.setZero();

  Eigen::MatrixXd W(tracking_result.previous_keypoints.size() * 2, tracking_result.previous_keypoints.size() * 2);
  W.setZero();

#pragma omp parallel for
  for (int kp_counter = 0; kp_counter < tracking_result.previous_keypoints.size(); kp_counter++)
  {
    const cv::KeyPoint &prev_kp = tracking_result.previous_keypoints[kp_counter];
    const cv::KeyPoint &cur_kp = tracking_result.current_keypoints[kp_counter];
    const float &tracking_error = tracking_result.tracking_error[kp_counter];

    // 1. Determine 3D points from previous frame
    std::optional<Eigen::Vector3d> opt_point_3d_prev = Get3DPoint(prev_kp);

    if (opt_point_3d_prev.has_value())
    {
      const Eigen::Vector3d &point_3d_prev = opt_point_3d_prev.value();
      // 2. Build Matrix
      int row_index = kp_counter * 2;
      // first row
      A(row_index, 0) = point_3d_prev.x();
      A(row_index, 1) = point_3d_prev.y();
      A(row_index, 2) = point_3d_prev.z();
      A(row_index, 3) = 1.0;
      A(row_index, 8) = -cur_kp.pt.x * point_3d_prev.x();
      A(row_index, 9) = -cur_kp.pt.x * point_3d_prev.y();
      A(row_index, 10) = -cur_kp.pt.x * point_3d_prev.z();
      A(row_index, 11) = -cur_kp.pt.x;
      // second row
      A(row_index + 1, 4) = -point_3d_prev.x();
      A(row_index + 1, 5) = -point_3d_prev.y();
      A(row_index + 1, 6) = -point_3d_prev.z();
      A(row_index + 1, 7) = -1.0;
      A(row_index + 1, 8) = cur_kp.pt.y * point_3d_prev.x();
      A(row_index + 1, 9) = cur_kp.pt.y * point_3d_prev.y();
      A(row_index + 1, 10) = cur_kp.pt.y * point_3d_prev.z();
      A(row_index + 1, 11) = cur_kp.pt.y;

      // 3. Set the weight matrix
      W(row_index, row_index) = DetermineWeight(tracking_error);
      W(row_index + 1, row_index + 1) = W(row_index, row_index);
    }
  }

  // std::cout << "A Matrix Size: " << A.rows() << "x" << A.cols() << std::endl;
  // std::cout << "W Matrix Size: " << W.rows() << "x" << W.cols() << std::endl;
  Eigen::MatrixXd M = A.transpose() * W * A;
  // std::cout << "M Matrix Size: " << M.rows() << "x" << M.cols() << std::endl;

  // 4. determine SVD-decomposition
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
  /*std::cout << "U: " << std::endl
            << svd.matrixU() << std::endl;
  std::cout << "Singular Values: " << std::endl
            << svd.singularValues() << std::endl;
  std::cout << "V: " << std::endl
            << svd.matrixV() << std::endl;*/

  // 5. Select singular vector to smalles singular vector
  Eigen::VectorXd p_scaled = svd.matrixV().col(11);

  // 6. Decompose transformation from projection vector
  const double &fx_rgb = calibration_params_.rgb_cam.at<double>(0, 0);
  const double &fy_rgb = calibration_params_.rgb_cam.at<double>(1, 1);
  const double &cx_rgb = calibration_params_.rgb_cam.at<double>(0, 2);
  const double &cy_rgb = calibration_params_.rgb_cam.at<double>(1, 2);

  Eigen::Vector3d translation_scaled{0.0, 0.0, 0.0};
  Eigen::Matrix3d rotation_scaled{};
  translation_scaled.z() = p_scaled(11);
  translation_scaled.x() = (p_scaled(3) - cx_rgb * translation_scaled.z()) / fx_rgb;
  translation_scaled.y() = (p_scaled(7) - cy_rgb * translation_scaled.z()) / fy_rgb;
  rotation_scaled(2, 0) = p_scaled(8);
  rotation_scaled(2, 1) = p_scaled(9);
  rotation_scaled(2, 2) = p_scaled(10);
  rotation_scaled(0, 0) = (p_scaled(0) - cx_rgb * rotation_scaled(2, 0)) / fx_rgb;
  rotation_scaled(0, 1) = (p_scaled(1) - cx_rgb * rotation_scaled(2, 1)) / fx_rgb;
  rotation_scaled(0, 2) = (p_scaled(2) - cx_rgb * rotation_scaled(2, 2)) / fx_rgb;
  rotation_scaled(1, 0) = (p_scaled(4) - cy_rgb * rotation_scaled(2, 0)) / fy_rgb;
  rotation_scaled(1, 1) = (p_scaled(5) - cy_rgb * rotation_scaled(2, 1)) / fy_rgb;
  rotation_scaled(1, 2) = (p_scaled(6) - cy_rgb * rotation_scaled(2, 2)) / fy_rgb;

  Eigen::JacobiSVD<Eigen::Matrix3d> svd_rot(rotation_scaled, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::Matrix3d rotation_orthogonal = svd_rot.matrixU() * svd_rot.matrixV().transpose();
  double scale = svd_rot.singularValues()(1); // get the median

  Eigen::VectorXd p = 1.0 / scale * p_scaled;
  Eigen::Vector3d translation{0.0, 0.0, 0.0};
  translation.z() = p(11);
  translation.x() = (p(3) - cx_rgb * translation_scaled.z()) / fx_rgb;
  translation.y() = (p(7) - cy_rgb * translation_scaled.z()) / fy_rgb;

  Eigen::Affine3d cur_T_prev;
  cur_T_prev.translation() = translation;
  cur_T_prev.linear() = rotation_orthogonal;

  Eigen::Affine3d prev_T_cur = cur_T_prev.inverse();

  return prev_T_cur;
}

cv::Mat RGBDSlam::DrawKeypoints(const cv::Mat &rgb_im)
{
  cv::Mat keypoints_im;
  rgb_im.copyTo(keypoints_im);
  for (int i = 0; i < tracked_keypoints_cpu_.size(); i++)
  {
    // draw detected keypoint
    int radiusCircle = 5;
    int thickness = 3;
    const cv::Point2f &kp = tracked_keypoints_cpu_[i].pt;
    cv::circle(keypoints_im, kp, radiusCircle, tracked_feature_color_[i], thickness);
  }
  return keypoints_im;
}

void RGBDSlam::Track(const double &timestamp, const cv::Mat &rgb_im, const cv::Mat &depth_im)
{
  std::cout << "Timestamp:\t" << std::setprecision(15) << timestamp << std::endl;

  // upload images to GPU
  rgb_im_.upload(rgb_im);
  cv::cuda::cvtColor(rgb_im_, rgb_im_, cv::COLOR_BGR2GRAY);
  depth_im.copyTo(depth_im_);

  // detect or track features
  TrackingResult tracking_result = DetectTrackFeatures();

  // estimate relative pose between previous and current time
  Eigen::Affine3d prev_T_cur = EstimateRelativePose(tracking_result);

  if (!std::isnan(prev_T_cur.matrix()(0, 3)))
  {
    start_T_cur = start_T_cur * prev_T_cur;
  }
  std::cout << start_T_cur.translation() << std::endl
            << "---" << std::endl;

  // last actions
  rgb_im_.copyTo(rgb_im_prev_);
  depth_im_.copyTo(depth_im_prev_);

  // show the image
  cv::Mat keypoint_rgb_im = DrawKeypoints(rgb_im);
  cv::imshow("depth image", depth_im);
  cv::imshow("rgb image", keypoint_rgb_im);
  cv::waitKey(1);
}

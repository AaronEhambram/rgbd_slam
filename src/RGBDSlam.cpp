#include "rgbd_slam/RGBDSlam.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iomanip>

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

void RGBDSlam::DetectTrackFeatures()
{
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
    std::vector<cv::KeyPoint> tracked_keypoints;
    std::vector<cv::Scalar> tracked_feature_color;
    tracked_keypoints.reserve(tracked_keypoints_cpu_.size());
    tracked_feature_color.reserve(tracked_feature_color_.size());
    cv::Mat tracked_descriptors;
    for (int i = 0; i < tracking_status_cpu.cols; ++i)
    {
      if ((int)tracking_status_cpu_col[i] == 1 && err_cpu_col[i] <= slam_params_.max_optical_flow_error)
      {
        const cv::Vec2f &f = features_new_cpu_col[i];
        // update the keypoint position
        tracked_keypoints_cpu_[i].pt = {f(0), f(1)};
        tracked_keypoints.emplace_back(tracked_keypoints_cpu_[i]);
        tracked_descriptors.push_back(tracked_descriptors_cpu_.row(i));
        tracked_feature_color.emplace_back(tracked_feature_color_[i]);
      }
    }

    // delete the untracked features by overwriting
    tracked_keypoints_cpu_ = tracked_keypoints;
    tracked_feature_color_ = tracked_feature_color;
    tracked_descriptors_cpu_ = cv::Mat(tracked_descriptors.size(), tracked_descriptors.type());
    tracked_descriptors.copyTo(tracked_descriptors_cpu_);

    // Identify masks with too less keypoints
    std::vector<int> features_in_mask(detection_masks_cpu_.size(), 0);
    for (const cv::KeyPoint& kp : tracked_keypoints_cpu_)
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
    for(int i = 0; i < features_in_mask.size(); i++)
    {
      if(features_in_mask[i] < slam_params_.min_features_in_mask_to_detect_new)
      {
        new_detection_mask_indices.emplace_back(i);
      }
    }

    // detect the new features in the underrepresented masks
    DetectNewFeatures(new_detection_mask_indices);
  }

  std::cout << tracked_keypoints_cpu_.size() << std::endl;
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
  depth_im_.upload(depth_im);
  cv::cuda::cvtColor(rgb_im_, rgb_im_, cv::COLOR_BGR2GRAY);

  // detect or track features
  DetectTrackFeatures();

  // last actions
  rgb_im_.copyTo(rgb_im_prev_);

  // show the image
  cv::Mat keypoint_rgb_im = DrawKeypoints(rgb_im);
  cv::imshow("depth image", depth_im);
  cv::imshow("rgb image", keypoint_rgb_im);
  cv::waitKey(1);
}
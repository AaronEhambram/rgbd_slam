#include "rgbd_slam/DepthImageOdometry.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <omp.h>

DepthImageOdometry::DepthImageOdometry(const CalibartionParameters &calibration_params) : calibration_params_{calibration_params}
{
  start_T_cur_ = Eigen::Isometry3d::Identity();
}

void DepthImageOdometry::Track(const double &timestamp, const cv::Mat &rgb_im, const cv::Mat &depth_im)
{
  if (prev_depth_im_.size() == depth_im.size())
  {
    // std::cout << "sum: " << DetermineDepthError(prev_depth_im_, depth_im, Eigen::Isometry3d::Identity()) << std::endl
    //           << "---" << std::endl;
    // std::cout << NumericJacobian(prev_depth_im_, depth_im, Eigen::Isometry3d::Identity()) << std::endl
    //          << "---" << std::endl;
    start_T_cur_ = start_T_cur_ * OptimizePose(prev_depth_im_, depth_im, Eigen::Isometry3d::Identity());
    std::cout << start_T_cur_.matrix() << std::endl
              << "---" << std::endl
              << std::endl;

    // Visualize
    cv::Mat depth_im_scaled;
    depth_im.convertTo(depth_im_scaled, CV_32F, 0.0001, 0);
    cv::imshow("depth image", depth_im_scaled);
    cv::waitKey(1);
  }

  // Finishing operations
  depth_im.copyTo(prev_depth_im_);
}

double DepthImageOdometry::DetermineDepthError(const cv::Mat &prev_depth_im, const cv::Mat &cur_depth_im, const Eigen::Isometry3d &prev_T_cur) const
{
  static const double &fx_depth = calibration_params_.depth_cam.at<double>(0, 0);
  static const double &fy_depth = calibration_params_.depth_cam.at<double>(1, 1);
  static const double &cx_depth = calibration_params_.depth_cam.at<double>(0, 2);
  static const double &cy_depth = calibration_params_.depth_cam.at<double>(1, 2);
  double error_sum = 0.0;
  const Eigen::Isometry3d cur_T_prev = prev_T_cur.inverse();
  int counter = 0;
#pragma omp parallel reduction(+ : error_sum, counter)
  {
#pragma omp for collapse(2)
    for (int v_prev = 0; v_prev < prev_depth_im.rows; v_prev++)
    {
      for (int u_prev = 0; u_prev < prev_depth_im.cols; u_prev++)
      {
        // 1. Determine 3D point in previous frame
        const unsigned short &depth_prev = prev_depth_im.at<unsigned short>(v_prev, u_prev);
        if (depth_prev == 0U)
        {
          // Does not contribute to error
          error_sum += 0.0;
          continue;
        }
        Eigen::Vector3d p_prev;
        p_prev << static_cast<double>(depth_prev) * (static_cast<double>(u_prev) - cx_depth) / fx_depth,
            static_cast<double>(depth_prev) * (static_cast<double>(v_prev) - cy_depth) / fy_depth,
            static_cast<double>(depth_prev);
        p_prev = p_prev * 0.001; // mm to m!

        // 2. transform point to current frame
        Eigen::Vector3d p_cur = cur_T_prev * p_prev;

        // 3. project to current depth image -> provides pixel points
        const double &depth_cur_expected = p_cur.z();
        if (std::abs(depth_cur_expected) < 1e-10)
        {
          // Does not contribute to error
          error_sum += 0.0;
          continue;
        }
        const double u_cur_subpixel = (fx_depth * p_cur.x() / depth_cur_expected + cx_depth);
        const double v_cur_subpixel = (fy_depth * p_cur.y() / depth_cur_expected + cy_depth);

        const double depth_cur_measured = InterpolateDepth(cur_depth_im, u_cur_subpixel, v_cur_subpixel);

        // 4. get the depth measured depth value at the pixel position
        if (depth_cur_measured > 0.0)
        {
          double depth_error = pow((depth_cur_measured * 0.001 - depth_cur_expected), 2.0);
          error_sum += (depth_error);
          counter++;
        }
        else
        {
          error_sum += 0.0;
          continue;
        }
      }
    }
  }
  return error_sum / static_cast<double>(counter);
}

Eigen::Matrix<double, 1, 6> DepthImageOdometry::NumericJacobian(const cv::Mat &prev_depth_im, const cv::Mat &cur_depth_im, const Eigen::Isometry3d &prev_T_cur) const
{
  // determine minimal parametric representation of the pose
  Eigen::Vector<double, 6> prev_xi_cur = ToPoseVector(prev_T_cur);

  // iterate through each dimension and compute differences
  Eigen::Vector<double, 6> diff_step;
  diff_step(0) = 2e-3;
  diff_step(1) = 2e-3;
  diff_step(2) = 1e-3;
  diff_step(3) = 1e-6;
  diff_step(4) = 1e-6;
  diff_step(5) = 1e-6;
  Eigen::Matrix<double, 1, 6> J{};
  for (int i = 0; i < 6; ++i)
  {
    Eigen::Vector<double, 6> prev_xi_cur_low{prev_xi_cur};
    prev_xi_cur_low(i) -= diff_step(i);
    Eigen::Vector<double, 6> prev_xi_cur_high{prev_xi_cur};
    prev_xi_cur_high(i) += diff_step(i);
    // std::cout << ToIsometry(prev_xi_cur_low).matrix() << std::endl
    //           << std::endl;
    const double error_low = DetermineDepthError(prev_depth_im, cur_depth_im, ToIsometry(prev_xi_cur_low));
    // std::cout << "error_low = " << error_low << std::endl;
    //  std::cout << ToIsometry(prev_xi_cur_high).matrix() << std::endl
    //            << std::endl;
    const double error_high = DetermineDepthError(prev_depth_im, cur_depth_im, ToIsometry(prev_xi_cur_high));
    // std::cout << "error_high = " << error_high << std::endl;
    J(0, i) = ((error_high - error_low) / (2.0 * diff_step(i)));
  }
  return J;
}

Eigen::Isometry3d DepthImageOdometry::OptimizePose(const cv::Mat &prev_depth_im, const cv::Mat &cur_depth_im, const Eigen::Isometry3d &prev_T_cur) const
{
  int iterations = 0;
  double v = 2.0;
  double epsilon1 = 1e-12;
  double epsilon2 = 1e-6;
  double tau = 1e-12;
  double roh = -1.0;
  Eigen::Isometry3d pose_isometry = prev_T_cur;
  Eigen::Isometry3d new_pose_isometry{};
  Eigen::Vector<double, 6> pose_vector = ToPoseVector(prev_T_cur);
  Eigen::Vector<double, 6> new_pose_vector{};
  Eigen::Vector<double, 6> pose_vector_delta{};
  Eigen::Matrix<double, 1, 6> J = NumericJacobian(prev_depth_im, cur_depth_im, pose_isometry);
  Eigen::Matrix<double, 6, 6> A = J.transpose() * J;
  double mu = tau * A.diagonal().maxCoeff();
  Eigen::Matrix<double, 6, 6> M{};
  double error = -DetermineDepthError(prev_depth_im, cur_depth_im, pose_isometry);
  double new_error{0.0};
  Eigen::Vector<double, 6> g = J.transpose() * error;
  bool stop = (g.lpNorm<Eigen::Infinity>() <= epsilon1);
  while (iterations < 20)
  {
    iterations++;
    roh = -1.0;
    while (!stop && roh <= 0.0)
    {
      M = A + mu * Eigen::Matrix<double, 6, 6>::Identity();
      pose_vector_delta = M.householderQr().solve(g);
      if (pose_vector_delta.norm() <= epsilon2)
      {
        stop = true;
      }
      else
      {
        new_pose_vector = pose_vector + pose_vector_delta;
        new_pose_isometry = ToIsometry(new_pose_vector);
        new_error = -DetermineDepthError(prev_depth_im, cur_depth_im, new_pose_isometry);
        roh = (pow(error, 2.0) - pow(new_error, 2.0)) / (pose_vector_delta.transpose() * (mu * pose_vector_delta + g));
        if (roh > 0.0)
        {
          // take update
          pose_vector = new_pose_vector;
          pose_isometry = new_pose_isometry;
          J = NumericJacobian(prev_depth_im, cur_depth_im, pose_isometry);
          A = J.transpose() * J;
          error = -DetermineDepthError(prev_depth_im, cur_depth_im, pose_isometry);
          g = J.transpose() * error;
          stop = (g.lpNorm<Eigen::Infinity>() <= epsilon1);
          mu = mu * std::max(1.0 / 3.0, 1 - pow((2.0 * roh - 1), 3.0));
          v = 2.0;
        }
        else
        {
          // increase damping
          mu *= v;
          v *= 2.0;
        }
      }
    }
  }
  return pose_isometry;
}

double DepthImageOdometry::InterpolateDepth(const cv::Mat &depth_im, const double &u, const double &v) const
{
  // region based bilinear interpolation
  const double u_floor = floor(u);
  const double u_ceil = ceil(u);
  const double v_floor = floor(v);
  const double v_ceil = ceil(v);
  if (u_floor >= 0 && u_ceil >= 0 && v_floor >= 0 && v_ceil >= 0 &&
      u_floor < depth_im.cols && u_ceil < depth_im.cols && v_floor < depth_im.rows && v_ceil < depth_im.rows)
  {
    const double d11 = static_cast<double>(depth_im.at<unsigned short>(v_floor, u_floor));
    const double d12 = static_cast<double>(depth_im.at<unsigned short>(v_ceil, u_floor));
    const double d21 = static_cast<double>(depth_im.at<unsigned short>(v_floor, u_ceil));
    const double d22 = static_cast<double>(depth_im.at<unsigned short>(v_ceil, u_ceil));
    if (d11 > 0.0 && d12 > 0.0 && d21 > 0.0 && d22 > 0.0)
    {
      const double denominator = ((u_ceil - u_floor) * (v_ceil - v_floor));
      const double w11 = (u_ceil - u) * (v_ceil - v) / denominator;
      const double w12 = (u_ceil - u) * (v - v_floor) / denominator;
      const double w21 = (u - u_floor) * (v_ceil - v) / denominator;
      const double w22 = (u - u_floor) * (v - v_floor) / denominator;
      return w11 * d11 + w12 * d12 + w21 * d21 + w22 * d22;
    }
  }

  // simple rounding
  const int u_rounded = static_cast<int>(round(u));
  const int v_rounded = static_cast<int>(round(v));
  if (u_rounded >= 0 && v_rounded >= 0 && u_rounded < depth_im.cols && v_rounded < depth_im.rows)
  {
    const unsigned short &depth_rounded_measured = depth_im.at<unsigned short>(v_rounded, u_rounded);
    return static_cast<double>(depth_rounded_measured);
  }
  return 0.0;
}

Eigen::Vector<double, 6> DepthImageOdometry::ToPoseVector(const Eigen::Isometry3d &pose_isometry) const
{
  const Eigen::AngleAxisd aa(pose_isometry.linear());
  const Eigen::Vector3d aa_rotation = aa.axis() * aa.angle();
  Eigen::Vector<double, 6> pose_vector;
  pose_vector.head<3>() = pose_isometry.translation();
  pose_vector.tail<3>() = aa_rotation;
  return pose_vector;
}

Eigen::Isometry3d DepthImageOdometry::ToIsometry(const Eigen::Vector<double, 6> &pose_vector) const
{
  const double angle = pose_vector.tail<3>().norm();
  const Eigen::Vector3d axis = pose_vector.tail<3>().normalized();
  Eigen::AngleAxisd aa{angle, axis};
  Eigen::Isometry3d pose_isometry{};
  pose_isometry.translation() = pose_vector.head<3>();
  pose_isometry.linear() = aa.toRotationMatrix();
  return pose_isometry;
}

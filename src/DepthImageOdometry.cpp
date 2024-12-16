#include "rgbd_slam/DepthImageOdometry.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <omp.h>

DepthImageOdometry::DepthImageOdometry(const CalibartionParameters &calibration_params) : calibration_params_{calibration_params}
{
}

void DepthImageOdometry::Track(const double &timestamp, const cv::Mat &rgb_im, const cv::Mat &depth_im)
{
  if (prev_depth_im_.size() == depth_im.size())
  {
    std::cout << NumericJacobian(prev_depth_im_, depth_im, Eigen::Isometry3d::Identity()) << std::endl;

    // Visualize
    cv::Mat depth_im_scaled;
    cv::Mat diff;
    cv::absdiff(depth_im, prev_depth_im_, diff);
    diff.convertTo(depth_im_scaled, CV_32F, 0.0001, 0);
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
#pragma omp parallel reduction(+ : error_sum)
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
        const int u_cur = static_cast<int>(round(fx_depth * p_cur.x() + cx_depth * depth_cur_expected));
        const int v_cur = static_cast<int>(round(fx_depth * p_cur.y() + cy_depth * depth_cur_expected));

        // 4. get the depth measured depth value at the pixel position
        if (u_cur >= 0 && v_cur >= 0 && u_cur < cur_depth_im.cols && v_cur < cur_depth_im.rows)
        {
          const unsigned short &depth_cur_measured = cur_depth_im.at<unsigned short>(v_prev, u_prev);
          if (depth_cur_measured > 0)
          {
            error_sum += pow((static_cast<double>(depth_cur_measured) * 0.001 - depth_cur_expected), 2.0);
          }
        }
        else
        {
          error_sum += 0.0;
        }
      }
    }
  }
  return error_sum;
}

Eigen::Matrix<double, 1, 6> DepthImageOdometry::NumericJacobian(const cv::Mat &prev_depth_im, const cv::Mat &cur_depth_im, const Eigen::Isometry3d &prev_T_cur) const
{
  // determine minimal parametric representation of the pose
  Eigen::Vector<double, 6> prev_xi_cur = ToPoseVector(prev_T_cur);

  // iterate through each dimension and compute differences
  double diff_step = 1e-4;
  Eigen::Matrix<double, 1, 6> J{};
  for(int i = 0; i < 6; ++i)
  {
    Eigen::Vector<double, 6> prev_xi_cur_low{prev_xi_cur};
    prev_xi_cur_low(i) -= diff_step;
    Eigen::Vector<double, 6> prev_xi_cur_high{prev_xi_cur};
    prev_xi_cur_low(i) += diff_step;
    const double error_low = DetermineDepthError(prev_depth_im, cur_depth_im, ToIsometry(prev_xi_cur_low));
    const double error_high = DetermineDepthError(prev_depth_im, cur_depth_im, ToIsometry(prev_xi_cur_high));
    J(0,i) = ((error_high - error_low)/(2.0*diff_step));
  }

  return J;
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

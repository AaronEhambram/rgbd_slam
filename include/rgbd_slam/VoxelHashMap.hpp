#ifndef VOXELHASHMAP_H_
#define VOXELHASHMAP_H_
#include "tsl/robin_map.h"
#include "Eigen/Dense"

struct Voxel
{
  std::vector<Eigen::Vector3d> points_;
};

struct VoxelHashFunction
{
  size_t operator()(const Eigen::Vector3i &cell) const 
  {
    const uint32_t *vec = reinterpret_cast<const uint32_t *>(cell.data());
    return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349663 ^ vec[2] * 83492791);
  }
};

class VoxelHashMap
{
  public:
  VoxelHashMap(const double& voxel_size, const double& max_points_per_voxel);
  VoxelHashMap(){};
  void add_points(const std::vector<Eigen::Vector3d>& points);
  void add_transformed_points(const Eigen::Affine3d& map_T_cur, const std::vector<Eigen::Vector3d>& points);
  std::vector<Eigen::Vector3d> get_voxelized_points();
  void get_correspondences(const Eigen::Affine3d& map_T_cur_predicted, const std::vector<Eigen::Vector3d>& points, std::vector<std::pair<size_t,Voxel*>>& correspondences);

  private:
  double voxel_size_;
  double max_points_per_voxel_; 
  tsl::robin_map<Eigen::Vector3i,Voxel,VoxelHashFunction> map_;
  std::vector<Eigen::Vector3d> voxelized_points_; 
};

#endif
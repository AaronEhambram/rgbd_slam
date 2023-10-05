#include "rgbd_slam/VoxelHashMap.hpp"
#include <iostream>

VoxelHashMap::VoxelHashMap(const double& voxel_size, const double& max_points_per_voxel)
{
  voxel_size_ = voxel_size;
  max_points_per_voxel_ = max_points_per_voxel;
  voxelized_points_.clear();
  map_.clear();
}

void VoxelHashMap::add_points(const std::vector<Eigen::Vector3d>& points)
{
  for(const Eigen::Vector3d& p : points)
  {
    Eigen::Vector3i grid_coord((p/voxel_size_).template cast<int>());
    tsl::robin_map<Eigen::Vector3i,Voxel,VoxelHashFunction>::iterator search_it = map_.find(grid_coord);
    if(search_it != map_.end())
    {
      // under the grid_coord a voxel is already there!
      // insert the point to the voxel, if less than max_points_per_voxel_ in the voxel
      if(search_it.value().points_.size() <= max_points_per_voxel_)
      {
        search_it.value().points_.push_back(p);
      }
    }
    else
    {
      // no Voxel at this cooordinate
      // create new voxel and save the point to the voxel
      map_[grid_coord].points_.push_back(p);
      voxelized_points_.push_back(p);
    }
  }
}

void VoxelHashMap::add_transformed_points(const Eigen::Affine3d& map_T_cur, const std::vector<Eigen::Vector3d>& points)
{
  std::vector<Eigen::Vector3d> transformed_points(points.size());
  #pragma omp parallel for
  for(int i = 0; i < points.size(); i++)
  {
    transformed_points[i] = map_T_cur*points[i]; 
  }
  add_points(transformed_points);
}

std::vector<Eigen::Vector3d> VoxelHashMap::get_voxelized_points()
{
  return voxelized_points_; 
}

void VoxelHashMap::get_correspondences(const Eigen::Affine3d& map_T_cur_predicted, const std::vector<Eigen::Vector3d>& points, std::vector<std::pair<size_t,Voxel*>>& correspondences)
{
  std::vector<Voxel*> closest_voxels(points.size(),NULL);
  #pragma omp parallel for
  for(int i = 0; i < points.size(); i++)
  {
    // transform points to map with predicted pose
    Eigen::Vector3d map_p = map_T_cur_predicted*points[i]; 
    Eigen::Vector3i map_p_grid_coord((map_p/voxel_size_).template cast<int>());
    double r = 0.15; // maximal radius
    int r_in_voxel = (int)std::ceil(r/voxel_size_); // radius in voxel_size

    // compute potential matches
    std::vector<Voxel*> potential_matches;
    potential_matches.reserve(pow(2.0*r_in_voxel,3.0));
    for(int x = map_p_grid_coord(0)-r_in_voxel; x < map_p_grid_coord(0)+r_in_voxel; x++)
    {
      for(int y = map_p_grid_coord(1)-r_in_voxel; y < map_p_grid_coord(1)+r_in_voxel; y++)
      {
        for(int z = map_p_grid_coord(2)-r_in_voxel; z < map_p_grid_coord(2)+r_in_voxel; z++)
        {
          Eigen::Vector3i grid_coord(x,y,z);
          // search in map 
          tsl::robin_map<Eigen::Vector3i,Voxel,VoxelHashFunction>::iterator search_it = map_.find(grid_coord);
          if(search_it != map_.end())
          {
            // insert the point to the voxel
            potential_matches.emplace_back(&search_it.value());
          }
        }
      }
    }
    // get nearest neighbor
    if(potential_matches.size() > 0)
    {
      Voxel* nearest_v = potential_matches[0]; 
      for(Voxel* v : potential_matches)
      {
        if((map_p - v->points_[0]).norm() < (map_p - nearest_v->points_[0]).norm())
        {
          nearest_v = v; 
        }
      }
      closest_voxels[i] = nearest_v;
    }
  }
  // copy the data to output
  for(int i = 0; i < closest_voxels.size(); i++)
  {
    if(closest_voxels[i] != NULL)
    {
      std::pair<size_t,Voxel*> pair(i,closest_voxels[i]);
      correspondences.push_back(pair);
    }
  }
}
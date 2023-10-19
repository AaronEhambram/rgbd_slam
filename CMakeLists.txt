cmake_minimum_required(VERSION 3.5)
project(rgbd_slam)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

SET(CMAKE_BUILD_TYPE Release)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(OpenCV_DIR "/home/ehambram/workspaces/libraries/opencv/opencv-4.x/install/lib/cmake/opencv4")
find_package(OpenCV 4.8 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(PCL REQUIRED)
set(TSL_INCLUDE_DIRS "/home/ehambram/workspaces/libraries/robin-map/include") # Header only library
set(g2o_DIR "/home/ehambram/workspaces/libraries/g2o/install/lib/cmake/g2o")
find_package(g2o REQUIRED)
set(G2O_USE_VENDORED_CERES true)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${TSL_INCLUDE_DIRS}
)

#add_executable(rgbd_slam_node src/rgbd_slam_node.cpp src/RGBDOdometry.cpp src/ICPOdometry.cpp src/VoxelHashMap.cpp)
#ament_target_dependencies(rgbd_slam_node rclcpp std_msgs cv_bridge sensor_msgs message_filters pcl_conversions geometry_msgs)
#target_link_libraries(rgbd_slam_node ${OpenCV_LIBS} OpenMP::OpenMP_CXX 
#                      g2o::core g2o::stuff g2o::g2o_cli_library  g2o::solver_cholmod g2o::solver_csparse g2o::csparse_extension
#                      g2o::solver_dense g2o::solver_pcg g2o::solver_slam2d_linear g2o::solver_structure_only g2o::solver_eigen
#                      g2o::types_icp g2o::types_sba g2o::types_sclam2d g2o::types_sim3 g2o::types_slam2d 
#                      g2o::types_slam3d)

# install the include folder
# install(DIRECTORY include/ DESTINATION include/${PROJECT_NAME})

#install(TARGETS
#  rgbd_slam_node
#  DESTINATION lib/${PROJECT_NAME}
#)

#install(DIRECTORY
#  launch
#  DESTINATION share/${PROJECT_NAME}/
#)

#ament_package()
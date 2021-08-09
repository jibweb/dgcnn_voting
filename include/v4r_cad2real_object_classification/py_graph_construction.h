#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/projection_matrix.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <v4r_cad2real_object_classification/parameters.h>
#include <v4r_cad2real_object_classification/scope_time.h>

typedef pcl::PointXYZINormal PointT;


class GraphConstructor
{
private:
  friend void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void* user_data_void);

protected:
  pcl::PointCloud<PointT>::Ptr pc_;
  std::vector<std::vector<uint> > adj_list_;
  Eigen::Vector4f centroid_;
  PointT min_pt_;
  PointT max_pt_;
  float voxel_size_;
  uint x_voxel_num_;
  uint y_voxel_num_;
  uint z_voxel_num_;

  // Parameters
  bool debug_;
  int lrf_code_;
  bool points_with_normals_;


  void voxelSeeding(std::unordered_map<int, std::tuple<uint, uint, float> > & voxels_map,
                         std::vector<uint> & pt_voxel_indices);
  void voxelAdjacency(std::unordered_map<int, std::tuple<uint, uint, float> > & voxels_map,
                       std::vector<std::vector<uint> > & voxels_adjacency,
                       std::vector<Eigen::Vector3f> & voxels_normals);
  void supervoxelRefinement(std::unordered_map<int, std::tuple<uint, uint, float> > & voxels_map,
                            std::vector<uint> & pt_voxel_indices,
                            std::vector<uint> & pt_supervoxel_indices,
                            std::vector<std::vector<uint> > & voxels_adjacency,
                            std::vector<std::unordered_set<uint> > & supervoxels_adjacency,
                            std::vector<Eigen::Vector3f> & voxels_normals,
                            std::vector<bool> & voxels_validity);

  // Internal graph processing
  void  normalAlignedPca(const pcl::IndicesPtr & indices, Eigen::Matrix3f & lrf, Eigen::Vector3f & mean);
  float regionScale(const pcl::IndicesPtr & indices, Eigen::Vector3f & mean);
  void searchSupportPointsNeighbors(std::vector<uint> & support_points,
                                    std::vector<std::vector<uint> > & neighbor_indices,
                                    int* neigh_indices, uint neighbors_nb,
                                    float shadowing_threshold, int* valid_indices);
  int sampleSupportPoints(std::vector<uint> & support_points,
                          std::vector<std::vector<uint> > & neighbor_indices,
                          std::vector<Eigen::Matrix3f> & lrf_transforms,
                          std::vector<float> & scales,
                          std::vector<Eigen::Vector3f> & means,
                          std::vector<std::vector<int> > & support_regions,
                          float neigh_size, int max_support_point, uint neighbors_nb, int seed);

  void normalSmoothing() {
    for (uint i=0; i<pc_->points.size(); i++) {
      Eigen::Vector3f normal = pc_->points[i].getNormalVector3fMap();
      for (auto neigh_idx : adj_list_[i])
        normal += pc_->points[neigh_idx].getNormalVector3fMap();

      normal.normalize();
      pc_->points[i].getNormalVector3fMap() = normal;
    }
  };
  void filterPoints(std::vector<bool> & points_to_keep);

  void updateCloudBbox() {
    pcl::compute3DCentroid (*pc_, centroid_);
    pcl::getMinMax3D (*pc_, min_pt_, max_pt_);

    x_voxel_num_ = static_cast<uint>((max_pt_.x - min_pt_.x) / voxel_size_) + 1;
    y_voxel_num_ = static_cast<uint>((max_pt_.y - min_pt_.y) / voxel_size_) + 1;
    z_voxel_num_ = static_cast<uint>((max_pt_.z - min_pt_.z) / voxel_size_) + 1;
  }


public:

  GraphConstructor(bool debug, int lrf, bool wnormals) :
    pc_(new pcl::PointCloud<PointT>),
    debug_(debug),
    lrf_code_(lrf),
    points_with_normals_(wnormals) {
      if (!debug_)
        pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
      else
        std::cout << "lrf_code: " << lrf << std::endl;

      voxel_size_ = 0.05f;
  }

  // Initialization
  void initializePLY(std::string filename, bool height_to_zero);
  void initializePCD(std::string filename, std::string filename_adj);
  void initializeScanObjectNN(std::string filename, std::string filename_adj, int filter_class_idx, bool height_to_zero);
  void initializeArray(float* vertices, uint vertex_nb, int* triangles, uint triangle_nb);
  void initializeArray(float* vertices, uint vertex_nb, int* triangles, uint triangle_nb, float* normals);

  void dataAugmentation(bool normal_smoothing, float normal_occlusion, float normal_noise);
  void getBbox(float* bbox) {
    // Center
    bbox[0] = centroid_(0);
    bbox[1] = centroid_(1);
    bbox[2] = centroid_(2);
    // Extent
    bbox[3] = max_pt_.x - min_pt_.x;
    bbox[4] = max_pt_.y - min_pt_.y;
    bbox[5] = max_pt_.z - min_pt_.z;
  };

  // Graph processing
  int sampleSupportPointsAndRegions(float* support_points_coords, int* neigh_indices, float* lrf_transforms,
                                    int* valid_indices, float* scales, int max_support_point, float neigh_size,
                                    int neighbors_nb, float shadowing_threshold, int seed, float** regions, uint region_sample_size,
                                    int disconnect_rate, float* heights);

  // Viz
  void vizGraph(int max_support_point, float neigh_size, uint neighbors_nb);
};

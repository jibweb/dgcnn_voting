#include <fstream>
#include <iterator>
#include <math.h>
#include <random>
#include <tuple>
#include <utility>


#include <pcl/common/distances.h>
#include <pcl/common/pca.h>
#include <pcl/conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>

#include <v4r_cad2real_object_classification/py_graph_construction.h>
#include <v4r_cad2real_object_classification/graph_visualization.h>

#include "graph_utils.cpp"
#include "augmentation_preprocessing.cpp"


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////// GENERAL ///////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::initializePLY(std::string filename, bool height_to_zero) {
  ScopeTime t("Graph Initialization (from PLY)", debug_);

  std::vector<std::array<uint32_t, 3> > triangles;
  tinyply::loadPLY(filename, *pc_, triangles);

  adj_list_.resize(pc_->points.size());

  for (auto t : triangles) {
    adj_list_[t[0]].push_back(t[1]);
    adj_list_[t[0]].push_back(t[2]);

    adj_list_[t[1]].push_back(t[0]);
    adj_list_[t[1]].push_back(t[2]);

    adj_list_[t[2]].push_back(t[0]);
    adj_list_[t[2]].push_back(t[1]);
  }

  if (height_to_zero) {
    float min_z = 10.;
    for (uint i=0; i<pc_->points.size(); i++) {
      if (pc_->points[i].z < min_z)
        min_z = pc_->points[i].z;
    }

    for (uint i=0; i<pc_->points.size(); i++) {
      pc_->points[i].z -= min_z;
    }
  }

  updateCloudBbox();

  if (debug_)
    std::cout << "PC size: " << pc_->points.size() << std::endl;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::initializePCD(std::string filename, std::string filename_adj) {
  ScopeTime t("Graph Initialization (from PCD with adj.)", debug_);

  if (pcl::io::loadPCDFile<PointT> (filename.c_str(), *pc_) == -1) {
    std::cout << "Couldn't read file " + filename + " \n" << std::endl;
  }

  adj_list_.resize(pc_->points.size());

  std::ifstream ss(filename_adj, std::ios::binary);
  std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(ss), {});
  int *puffer = (int*) &buffer[0];
  uint cur = 0, pt_idx = 0;

  int k = 0;
  while (cur < buffer.size()/sizeof(int)) {
    k = puffer[cur++];

    adj_list_[pt_idx].resize(k);
    memcpy(&adj_list_[pt_idx][0], &buffer[4*cur], 4*k);

    cur += k;
    pt_idx++;
  }

  updateCloudBbox();

  if (debug_)
    std::cout << "PC size: " << pc_->points.size() << std::endl;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::initializeScanObjectNN(std::string filename, std::string filename_adj, int filter_class_idx, bool height_to_zero) {
  ScopeTime t("Graph Initialization (from PCD with adj.)", debug_);

  std::ifstream pt_ss(filename, std::ios::binary);
  std::vector<unsigned char> pt_buffer(std::istreambuf_iterator<char>(pt_ss), {});
  float *pt_puffer = (float*) &pt_buffer[0];

  uint pc_len = static_cast<int>(pt_puffer[0]);
  pc_->points.resize(pc_len);

  std::vector<uint> label_count(41, 0);

  for (uint pt_idx=0; pt_idx<pc_len; pt_idx++) {
    pc_->points[pt_idx].x = pt_puffer[11*pt_idx + 1];
    pc_->points[pt_idx].z = pt_puffer[11*pt_idx + 2];
    pc_->points[pt_idx].y= pt_puffer[11*pt_idx + 3];

    pc_->points[pt_idx].normal_x = pt_puffer[11*pt_idx + 4];
    pc_->points[pt_idx].normal_z = pt_puffer[11*pt_idx + 5];
    pc_->points[pt_idx].normal_y= pt_puffer[11*pt_idx + 6];

    label_count[pt_puffer[11*pt_idx + 11]]++;

    // memcpy(&(pc_->points[pt_idx].data[0]), &pt_puffer[11*pt_idx + 1], 4*3);
    // memcpy(&(pc_->points[pt_idx].normal[0]), &pt_puffer[11*pt_idx + 4], 4*3);
  }

  adj_list_.resize(pc_->points.size());

  std::ifstream ss_adj(filename_adj, std::ios::binary);
  std::vector<unsigned char> adj_buffer(std::istreambuf_iterator<char>(ss_adj), {});
  int *puffer = (int*) &adj_buffer[0];
  uint cur = 0, pt_idx = 0;

  int k = 0;
  while (cur < adj_buffer.size()/sizeof(int)) {
    k = puffer[cur++];

    adj_list_[pt_idx].resize(k);
    memcpy(&adj_list_[pt_idx][0], &adj_buffer[4*cur], 4*k);

    cur += k;
    pt_idx++;
  }

  if (height_to_zero) {
    float min_z = 10.;
    for (uint i=0; i<pc_->points.size(); i++) {
      if (pc_->points[i].z < min_z)
        min_z = pc_->points[i].z;
    }

    for (uint i=0; i<pc_->points.size(); i++) {
      pc_->points[i].z -= min_z;
    }
  }

  updateCloudBbox();

  if (filter_class_idx != -1) {
    // Find the most represented label beside 0-unlabeled, 1-floor and 2-wall
    uint max_idx=0, max_cnt=0;
    for (uint i=3; i<label_count.size(); i++) {
      if (label_count[i] > max_cnt) {
        max_cnt = label_count[i];
        max_idx = i;
      }
    }

    // Create a mask of the points to keep
    std::vector<bool> non_bg(pc_->points.size(), false);
    for (uint pt_idx=0; pt_idx<pc_len; pt_idx++) {
      if (pt_puffer[11*pt_idx + 11] == max_idx) {
        non_bg[pt_idx] = true;
      }
    }

    filterPoints(non_bg);
  }

  if (debug_)
    std::cout << "PC size: " << pc_->points.size() << std::endl;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::initializeArray(float* vertices, uint vertex_nb,
                                       int* triangles, uint triangle_nb) {
  ScopeTime t("Normal Computation", debug_);

  // Normal Computation
  std::vector<Eigen::Vector3f> vertex_normals(vertex_nb, Eigen::Vector3f::Zero());
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> > vertices_map(vertices, vertex_nb, 3);

  for (uint tri_idx=0; tri_idx<triangle_nb; tri_idx++) {
    Eigen::Vector3f p0 = vertices_map.row(triangles[3*tri_idx + 0]);
    Eigen::Vector3f p1 = vertices_map.row(triangles[3*tri_idx + 1]);
    Eigen::Vector3f p2 = vertices_map.row(triangles[3*tri_idx + 2]);

    Eigen::Vector3f tri_normal = triangle_normal(p0, p1, p2);
    vertex_normals[triangles[3*tri_idx + 0]] += tri_normal;
    vertex_normals[triangles[3*tri_idx + 1]] += tri_normal;
    vertex_normals[triangles[3*tri_idx + 2]] += tri_normal;
  }


  for (uint pt_idx=0; pt_idx<vertex_nb; pt_idx++) {
    Eigen::Vector3f normal = vertex_normals[pt_idx];
    normal.normalize();

    vertex_normals[pt_idx] = normal;
  }

  // Initialize with the newly computed normals
  initializeArray(vertices, vertex_nb, triangles, triangle_nb, vertex_normals[0].data());
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::initializeArray(float* vertices, uint vertex_nb,
                                      int* triangles, uint triangle_nb,
                                      float* normals) {
  ScopeTime t("Graph Initialization (from arrays)", debug_);

  adj_list_.resize(vertex_nb);
  Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> > triangles_map(triangles, triangle_nb, 3);

  for (uint tri_idx=0; tri_idx < triangle_nb; tri_idx++) {
    Eigen::Vector3i t = triangles_map.row(tri_idx);

    adj_list_[t(0)].push_back(t(1));
    adj_list_[t(0)].push_back(t(2));

    adj_list_[t(1)].push_back(t(0));
    adj_list_[t(1)].push_back(t(2));

    adj_list_[t(2)].push_back(t(0));
    adj_list_[t(2)].push_back(t(1));
  }

  pc_->points.resize(vertex_nb);
  for (uint pt_idx=0; pt_idx < vertex_nb; pt_idx++) {
    pc_->points[pt_idx].x = vertices[3*pt_idx + 0];
    pc_->points[pt_idx].y = vertices[3*pt_idx + 1];
    pc_->points[pt_idx].z = vertices[3*pt_idx + 2];

    pc_->points[pt_idx].normal_x = normals[3*pt_idx + 0];
    pc_->points[pt_idx].normal_y = normals[3*pt_idx + 1];
    pc_->points[pt_idx].normal_z = normals[3*pt_idx + 2];
  }

  updateCloudBbox();

  if (debug_)
    std::cout << "PC size: " << pc_->points.size() << std::endl;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////// DATA AUGMENTATION ///////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::dataAugmentation(bool normal_smoothing, float normal_occlusion, float normal_noise) {
  ScopeTime t("Data Augmentation", debug_);

  bool z_rotation = false;
  if (z_rotation) {
    struct timeval time;
    gettimeofday(&time,NULL);
    srand((time.tv_sec * 1000) + (time.tv_usec / 1000));

    float rdn_angle = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(2.*M_PI)) );

    Eigen::Matrix4f rotation_z;
    rotation_z << cos(rdn_angle), -sin(rdn_angle), 0, 0,
                  sin(rdn_angle), cos(rdn_angle),  0, 0,
                  0,              0,               1, 0,
                  0,              0,               0, 1;

    pcl::transformPointCloudWithNormals(*pc_, *pc_, rotation_z);
    updateCloudBbox();
  }


  if (normal_smoothing)
    normalSmoothing();

  if (normal_occlusion > -1.f) {
    /*  Pick a random point and only keep points whose normal dot product to the
      reference point normal is higher than normal_occlusion.
      normal_occlusion set to -1 keeps all points, set to 1 rejects almost everything */
    ScopeTime t("Normal occlusion", debug_);

    std::vector<bool> points_to_keep(pc_->points.size(), false);

    uint n_idx = rand() % pc_->points.size();
    Eigen::Vector3f n_rnd = pc_->points[n_idx].getNormalVector3fMap();

    for (uint i=0; i<pc_->points.size(); i++) {
      if (n_rnd.dot(pc_->points[i].getNormalVector3fMap()) > normal_occlusion)
        points_to_keep[i] = true;
    }

    filterPoints(points_to_keep);
  }

  if (normal_noise > 0.f) {
    /*  Add a random amount between 0. and normal_noise to the x,y,z components
      of each normals and normalize them afterwards */
    ScopeTime t("Normal noise", debug_);

    float rdn_x, rdn_y, rdn_z;
    Eigen::Vector3f n_noise;

    for (uint i=0; i<pc_->points.size(); i++) {
      rdn_x = 2.f * static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/normal_noise)) - normal_noise;
      rdn_y = 2.f * static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/normal_noise)) - normal_noise;
      rdn_z = 2.f * static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/normal_noise)) - normal_noise;

      n_noise << rdn_x, rdn_y, rdn_z;
      pc_->points[i].getNormalVector3fMap() += n_noise;
      pc_->points[i].getNormalVector3fMap().normalize();
    }
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::filterPoints(std::vector<bool> & points_to_keep) {
  std::vector<uint> new_idx(pc_->points.size(), 0);
  uint valid_count = 0;

  for (uint pt_idx=0; pt_idx<pc_->points.size(); pt_idx++) {
    if (points_to_keep[pt_idx]) {
      new_idx[pt_idx] = valid_count;
      valid_count++;
    }
  }

  for (uint pt_idx=0; pt_idx<pc_->points.size(); pt_idx++) {
    if (!points_to_keep[pt_idx])
      continue;

    pc_->points[new_idx[pt_idx]] = pc_->points[pt_idx];

    std::vector<uint> filtered_neighbors;
    filtered_neighbors.reserve(adj_list_[pt_idx].size());

    for (auto neighbor : adj_list_[pt_idx]) {
      if (points_to_keep[neighbor])
        filtered_neighbors.push_back(new_idx[neighbor]);
    }

    adj_list_[new_idx[pt_idx]] = std::move(filtered_neighbors);
  }

  pc_->points.resize(valid_count);
  adj_list_.resize(valid_count);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////// GRAPH PROCESSING ////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int GraphConstructor::sampleSupportPointsAndRegions(float* support_points_coords, int* neigh_indices, float* lrf_transforms,
                                                    int* valid_indices, float* scales, int max_support_point, float neigh_size,
                                                    int neighbors_nb, float shadowing_threshold, int seed, float** regions, uint region_sample_size,
                                                    int disconnect_rate, float* heights) {

  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> > support_points_coords_map(support_points_coords, max_support_point, 3);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 9, Eigen::RowMajor> > lrf_transforms_map(lrf_transforms, max_support_point, 9);

  std::vector<uint> support_points;
  std::vector<std::vector<uint> > neighbor_indices;
  std::vector<Eigen::Matrix3f> lrf_transforms_vec;
  std::vector<float> scales_vec;
  std::vector<Eigen::Vector3f> means_vec;
  std::vector<std::vector<int> > support_regions;

  sampleSupportPoints(support_points, neighbor_indices, lrf_transforms_vec, scales_vec, means_vec, support_regions,
                      neigh_size, max_support_point, neighbors_nb, seed);

  std::cout << "Ok 1" << std::endl;

  for (uint pt_idx=0; pt_idx < support_points.size(); pt_idx++) {
    int random_draw = rand() % 100;
    if (random_draw < disconnect_rate)
      continue;

    // support_points_coords_map.row(pt_idx) = pc_->points[support_points[pt_idx]].getVector3fMap();
    support_points_coords_map.row(pt_idx) = means_vec[pt_idx];

    // !!! the LRF encoded in lrf_row is the transpose of the one in lrf_transforms_vec
    // lrf_row is ColumnMajor as is Eigen by default
    // lrf_transforms_map encode the transpose of the LRF computed (but that works out for SKPConv)
    Eigen::Map<Eigen::Matrix3f> lrf_row(lrf_transforms_map.row(pt_idx).data(), 3, 3);
    lrf_row = lrf_transforms_vec[pt_idx];

    valid_indices[pt_idx] = 1;
    scales[pt_idx] = scales_vec[pt_idx];
    heights[pt_idx] = means_vec[pt_idx](2) - min_pt_.z;
  }

  std::cout << "Ok 2" << std::endl;


  // searchSupportPointsNeighbors(support_points, neighbor_indices, neigh_indices, neighbors_nb, shadowing_threshold, valid_indices);

  if (points_with_normals_) {
    //TODO
  } else {
    for (uint support_pt_idx=0; support_pt_idx<support_points.size(); support_pt_idx++) {
      Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> > region_pts_map(regions[support_pt_idx], region_sample_size, 3);
      // Eigen::Vector3f support_center = pc_->points[support_points[support_pt_idx]].getVector3fMap();

      for (uint feat_idx=0; feat_idx < region_sample_size; feat_idx++) {
        uint rand_idx = rand() % support_regions[support_pt_idx].size();
        uint pt_idx = support_regions[support_pt_idx][rand_idx];
        Eigen::Vector3f centered_coords = pc_->points[pt_idx].getVector3fMap() - means_vec[support_pt_idx];
        // Eigen::Vector3f centered_coords = pc_->points[pt_idx].getVector3fMap() - support_center;

        region_pts_map.row(feat_idx) = scales[support_pt_idx] * lrf_transforms_vec[support_pt_idx] * centered_coords;
      }
    }
    std::cout << "Ok 3" << std::endl;

  }
}



void GraphConstructor::voxelSeeding(std::unordered_map<int, std::tuple<uint, uint, float> > & voxels_map,
                                         std::vector<uint> & pt_voxel_indices) {
  ScopeTime t("Voxel Seeding", debug_);

  float half_voxel_size = voxel_size_ / 2.f;

  uint voxel_num = 0;
  uint x_idx, y_idx, z_idx;
  int key;
  float dist_to_voxel_center;

  for (uint pt_idx=0; pt_idx < pc_->points.size(); pt_idx++) {
    x_idx = static_cast<uint>((pc_->points[pt_idx].x - min_pt_.x) / voxel_size_);
    y_idx = static_cast<uint>((pc_->points[pt_idx].y - min_pt_.y) / voxel_size_);
    z_idx = static_cast<uint>((pc_->points[pt_idx].z - min_pt_.z) / voxel_size_);

    PointT cur_voxel_center;
    cur_voxel_center.x = min_pt_.x + voxel_size_*x_idx + half_voxel_size;
    cur_voxel_center.y = min_pt_.y + voxel_size_*y_idx + half_voxel_size;
    cur_voxel_center.z = min_pt_.z + voxel_size_*z_idx + half_voxel_size;

    dist_to_voxel_center = pcl::squaredEuclideanDistance(pc_->points[pt_idx], cur_voxel_center);
    key = x_idx + y_idx*x_voxel_num_ + z_idx*x_voxel_num_*y_voxel_num_;

    auto search = voxels_map.find(key);
    if (search != voxels_map.end()) {
      // There is already an entry in the map for that voxel

      // Check the normal to separate the two sides of a thin object
      Eigen::Vector3f n_pt = pc_->points[pt_idx].getNormalVector3fMap();
      Eigen::Vector3f n_vox = pc_->points[std::get<1>(search->second)].getNormalVector3fMap();
      float vox_pt_angle = acosf(std::max(-1.0f, std::min(1.0f, (n_pt.dot(n_vox)))));


      if (vox_pt_angle > M_PI / 2.f) {
        search = voxels_map.find(-key);
        if (search != voxels_map.end()) {
          // There is already an entry for the opposite voxel
          pt_voxel_indices[pt_idx] = std::get<0>(search->second);

          if (dist_to_voxel_center < std::get<2>(search->second)) {
            std::get<2>(search->second) = dist_to_voxel_center;
            std::get<1>(search->second) = pt_idx;
          }
        } else {
          // There is no entry for the opposite voxel
          pt_voxel_indices[pt_idx] = voxel_num;
          voxels_map[-key] = std::make_tuple(voxel_num, pt_idx, dist_to_voxel_center);
          voxel_num++;
        }

      } else {
        // The current point fits the direction of the voxel
        pt_voxel_indices[pt_idx] = std::get<0>(search->second);

        if (dist_to_voxel_center < std::get<2>(search->second)) {
          std::get<2>(search->second) = dist_to_voxel_center;
          std::get<1>(search->second) = pt_idx;
        }
      }

    } else {
      // There no entry yet for that voxel
      pt_voxel_indices[pt_idx] = voxel_num;
      voxels_map[key] = std::make_tuple(voxel_num, pt_idx, dist_to_voxel_center);
      voxel_num++;
    }

  }
}


void GraphConstructor::voxelAdjacency(std::unordered_map<int, std::tuple<uint, uint, float> > & voxels_map,
                                      std::vector<std::vector<uint> > & voxels_adjacency,
                                      std::vector<Eigen::Vector3f> & voxels_normals) {
  ScopeTime t("Voxel Adjacency", debug_);

  uint voxel_num = voxels_map.size();
  std::vector<int> voxels_x(voxel_num);
  std::vector<int> voxels_y(voxel_num);
  std::vector<int> voxels_z(voxel_num);

  for (auto& v : voxels_map) {
    // Recover the original grid index
    int key = abs(v.first);
    int vx_idx = static_cast<int>(key % x_voxel_num_);
    int vyz = static_cast<int>(key / x_voxel_num_);
    int vy_idx = vyz % y_voxel_num_;
    int vz_idx = vyz / y_voxel_num_;
    uint voxel_idx = std::get<0>(v.second);

    voxels_x[voxel_idx] = vx_idx;
    voxels_y[voxel_idx] = vy_idx;
    voxels_z[voxel_idx] = vz_idx;
  }

  for (uint voxel_idx=0; voxel_idx<voxel_num; voxel_idx++)
    voxels_adjacency[voxel_idx].reserve(20);

  for (uint voxel_idx=0; voxel_idx<voxel_num; voxel_idx++) {
    int vx_idx = voxels_x[voxel_idx];
    int vy_idx = voxels_y[voxel_idx];
    int vz_idx = voxels_z[voxel_idx];

    for (int x_diff=-1; x_diff<2; x_diff++) {
      for (int y_diff=-1; y_diff<2; y_diff++) {
        for (int z_diff=-1; z_diff<2; z_diff++) {
          if ((x_diff==0) && (y_diff==0) && (z_diff==0))
            continue;

          if (((vx_idx + x_diff) < 0) ||
              ((vy_idx + y_diff) < 0) ||
              ((vz_idx + z_diff) < 0) ||
              ((vx_idx + x_diff) >= x_voxel_num_) ||
              ((vy_idx + y_diff) >= y_voxel_num_) ||
              ((vz_idx + z_diff) >= z_voxel_num_))
            continue;


          int key = (vx_idx + x_diff) + (vy_idx + y_diff)*x_voxel_num_ + (vz_idx + z_diff)*x_voxel_num_*y_voxel_num_;
          auto search = voxels_map.find(key);

          if (search != voxels_map.end()) {
            uint neigh_voxel_idx = std::get<0>(search->second);

            if ((acosf(std::max(-1.0f, std::min(1.0f, (voxels_normals[voxel_idx].dot(voxels_normals[neigh_voxel_idx]))))) < M_PI/2.f)) {
              voxels_adjacency[voxel_idx].push_back(neigh_voxel_idx);
            } else {
              search = voxels_map.find(-key);

              if (search != voxels_map.end()) {
                neigh_voxel_idx = std::get<0>(search->second);
                voxels_adjacency[voxel_idx].push_back(neigh_voxel_idx);
              }
            }
          }
        } // -- end for z_diff
      } // -- end for y_diff
    } // -- end for x_diff
  } //-- end for voxel_idx
}



void GraphConstructor::supervoxelRefinement(std::unordered_map<int, std::tuple<uint, uint, float> > & voxels_map,
                                            std::vector<uint> & pt_voxel_indices,
                                            std::vector<uint> & pt_supervoxel_indices,
                                            std::vector<std::vector<uint> > & voxels_adjacency,
                                            std::vector<std::unordered_set<uint> > & supervoxels_adjacency,
                                            std::vector<Eigen::Vector3f> & voxels_normals,
                                            std::vector<bool> & voxels_validity) {
  ScopeTime t("Supervoxel Refinement", debug_);

  std::vector<uint> pt_voxel_dist(pc_->points.size(), 1e4);
  std::vector<uint> voxel_to_supervoxel(voxels_map.size());
  uint x_idx, y_idx, z_idx;

  for (auto& v : voxels_map) {
    // Recover the original grid index
    uint voxel_idx = std::get<0>(v.second);

    int key = abs(v.first);
    int vx_idx = static_cast<int>(key % x_voxel_num_);
    int vyz = static_cast<int>(key / x_voxel_num_);
    int vy_idx = vyz % y_voxel_num_;
    int vz_idx = vyz / y_voxel_num_;

    Eigen::Vector3f n_vox = voxels_normals[voxel_idx];

    std::deque<uint> queue;
    uint sampled_idx = std::get<1>(v.second);
    Eigen::Vector3f n_sampled = pc_->points[sampled_idx].getNormalVector3fMap();
    float sampled_dist = acosf(std::max(-1.0f, std::min(1.0f, (n_vox.dot(n_sampled)))));
    if (sampled_dist < pt_voxel_dist[sampled_idx]) {
      pt_voxel_dist[sampled_idx] = sampled_dist;
      pt_supervoxel_indices[sampled_idx] = voxel_idx;
      queue.push_back(sampled_idx);
      voxels_validity[voxel_idx] = true;
    }

    std::vector<bool> visited(pc_->points.size(), false);
    visited[sampled_idx] = true;

    while(!queue.empty()) {
      // Dequeue a face
      uint pt_idx = queue.front();
      queue.pop_front();

      for (auto neigh_idx : adj_list_[pt_idx]) {
        if (visited[neigh_idx])
          continue;

        visited[neigh_idx] = true;

        if (pt_voxel_indices[neigh_idx] != voxel_idx) {
          x_idx = static_cast<uint>((pc_->points[neigh_idx].x - min_pt_.x) / voxel_size_);
          y_idx = static_cast<uint>((pc_->points[neigh_idx].y - min_pt_.y) / voxel_size_);
          z_idx = static_cast<uint>((pc_->points[neigh_idx].z - min_pt_.z) / voxel_size_);

          // Checking if the considered point is inside a neighboring voxel
          if ((abs(vx_idx - static_cast<int>(x_idx)) > 1) ||
              (abs(vy_idx - static_cast<int>(y_idx)) > 1) ||
              (abs(vz_idx - static_cast<int>(z_idx)) > 1))
            continue;
        }

        // The point is within the neighborhood of the voxel of interest
        Eigen::Vector3f n_neigh = pc_->points[neigh_idx].getNormalVector3fMap();
        float neigh_dist = acosf(std::max(-1.0f, std::min(1.0f, (n_vox.dot(n_neigh)))));

        if (neigh_dist < pt_voxel_dist[neigh_idx]) {
          pt_voxel_dist[neigh_idx] = neigh_dist;
          pt_supervoxel_indices[neigh_idx] = voxel_idx;
          queue.push_back(neigh_idx);
        }
      }
    } // -- end while(!queue.empty())

  } // -- end for v in voxels_map


  for (auto v : voxels_map) {
    voxel_to_supervoxel[std::get<0>(v.second)] = pt_supervoxel_indices[std::get<1>(v.second)];
  }


  for (uint i=0; i<pt_supervoxel_indices.size(); i++) {
    if (pt_supervoxel_indices[i] == 1e5)
      pt_supervoxel_indices[i] = voxel_to_supervoxel[pt_voxel_indices[i]];
  }


  for (uint voxel_idx=0; voxel_idx<voxels_map.size(); voxel_idx++)
    for (auto neigh_idx : voxels_adjacency[voxel_idx])
      supervoxels_adjacency[voxel_to_supervoxel[voxel_idx]].insert(voxel_to_supervoxel[neigh_idx]);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int GraphConstructor::sampleSupportPoints(std::vector<uint> & support_points,
                                          std::vector<std::vector<uint> > & neighbor_indices,
                                          std::vector<Eigen::Matrix3f> & lrf_transforms,
                                          std::vector<float> & scales,
                                          std::vector<Eigen::Vector3f> & means,
                                          std::vector<std::vector<int> > & support_regions,
                                          float neigh_size, int max_support_point, uint neighbors_nb, int seed) {

  //Pre-allocate memory
  support_points.reserve(max_support_point);
  lrf_transforms.reserve(max_support_point);
  scales.reserve(max_support_point);
  means.reserve(max_support_point);
  support_regions.reserve(max_support_point);

  // --- Initial seeds sampling and voxel assignment -------------------------------------------------------------
  std::unordered_map<int, std::tuple<uint, uint, float> > voxels_map;
  std::vector<uint> pt_voxel_indices(pc_->points.size());
  voxelSeeding(voxels_map, pt_voxel_indices);
  uint voxel_num = voxels_map.size();


  // --- Voxel normal computation --------------------------------------------------------------------------------
  std::vector<Eigen::Vector3f> voxels_normals(voxel_num);

  for (auto& v : voxels_map) {
    Eigen::Vector3f normal = pc_->points[std::get<1>(v.second)].getNormalVector3fMap();

    for (auto neigh_idx : adj_list_[std::get<1>(v.second)]) {
      normal += pc_->points[neigh_idx].getNormalVector3fMap();
    }

    normal.normalize();
    voxels_normals[std::get<0>(v.second)] = normal;
  }


  // --- Voxel adjacency -----------------------------------------------------------------------------------------
  std::vector<std::vector<uint> > voxels_adjacency(voxel_num);
  voxelAdjacency(voxels_map, voxels_adjacency, voxels_normals);


  // --- Voxel assignment refinment ------------------------------------------------------------------------------
  std::vector<uint> pt_supervoxel_indices(pc_->points.size(), 1e5);
  std::vector<bool> voxels_validity(voxel_num, false);
  std::vector<std::unordered_set<uint> > supervoxels_adjacency(voxel_num);
  supervoxelRefinement(voxels_map, pt_voxel_indices, pt_supervoxel_indices,
                       voxels_adjacency, supervoxels_adjacency,
                       voxels_normals, voxels_validity);


  uint valid_voxel_num = 0;
  // std::vector<int> voxel_to_supervoxel_indices(voxel_num, -1);
  for (uint voxel_idx=0; voxel_idx<voxel_num; voxel_idx++) {
    if (voxels_validity[voxel_idx]) {
      // voxel_to_supervoxel_indices[voxel_idx] = valid_voxel_num;
      valid_voxel_num++;
    }
  }

  std::vector<pcl::IndicesPtr> supervoxels_indices;
  supervoxels_indices.reserve(voxel_num);
  for( int i = 0; i < voxel_num; ++i ) {
    supervoxels_indices.emplace_back(new std::vector<int>());
    supervoxels_indices[i]->reserve(50);
  }

  for (uint pt_idx=0; pt_idx<pc_->points.size(); pt_idx++) {
    uint supervoxel_idx = pt_supervoxel_indices[pt_idx];
    supervoxels_indices[supervoxel_idx]->push_back(pt_idx);
  }

  std::vector<PointT> voxels_centers(voxel_num);
  for (auto v : voxels_map) {
    voxels_centers[std::get<0>(v.second)] = pc_->points[std::get<1>(v.second)];
  }


  std::cout << "Points: " << pc_->points.size() << " / Voxels: " << voxel_num << " / Valid voxels: " << valid_voxel_num << std::endl;


  std::vector<double> concavity_bels(voxel_num, 0.5);
  std::vector<double> convexity_bels(voxel_num, 0.5);

  std::vector<Eigen::Vector3f> supervoxels_centers(voxel_num);
  std::vector<Eigen::Vector3f> supervoxels_normals(voxel_num);

{

  for (uint voxel_idx=0; voxel_idx<voxel_num; voxel_idx++) {

    if (supervoxels_indices[voxel_idx]->size() <= 3)
      continue;

    Eigen::Vector3f mean;
    Eigen::Matrix3f lrf;
    normalAlignedPca(supervoxels_indices[voxel_idx], lrf, mean);
    supervoxels_centers[voxel_idx] = mean;
    supervoxels_normals[voxel_idx] = lrf.row(2);
  }



  ScopeTime t("Convexity", debug_);
  double local_convexity_conf, norm;

  for (uint pt_idx=0; pt_idx<pc_->points.size(); pt_idx++) {
    // Measure the convexity between the point and its corresponding voxel
    uint voxel_idx = pt_supervoxel_indices[pt_idx];
    local_convexity_conf = convexityConfidence(supervoxels_centers[voxel_idx],
                                               pc_->points[pt_idx].getVector3fMap(),
                                               supervoxels_normals[voxel_idx],
                                               pc_->points[pt_idx].getNormalVector3fMap());

    // Fuse that local measurement into a voxel-level prediction
    convexity_bels[voxel_idx] *= local_convexity_conf;
    concavity_bels[voxel_idx] *= 1. - local_convexity_conf;
    norm = concavity_bels[voxel_idx] + convexity_bels[voxel_idx];
    concavity_bels[voxel_idx] /= norm;
    convexity_bels[voxel_idx] /= norm;
  }
} // -- end ScopeTime t("Convexity", debug_);


  // std::vector<uint> pt_merged_voxel_indices(pc_->points.size(), 1e5);
  std::vector<std::vector<uint> > segments_to_supervoxels;
  std::vector<std::vector<uint> > supervoxels_to_segments;
  supervoxels_to_segments.resize(voxel_num);

{
  ScopeTime t("Segment Merging", debug_);
  // std::vector<uint> voxel_to_merged_voxel(voxel_num);
  segments_to_supervoxels.reserve(100);
  std::vector<bool> is_processed(voxel_num, false);
  double convexity_conf;

  // Maybe randomize the order of voxels
  // convexity average convexity instead ? Merge neighboring segment with similar average convexity ?
  // remove the acos
  // look at overlap
  // improve merging, for neater clusters. Double check same_proj as it looks weird right now

  for (uint voxel_idx=0; voxel_idx<voxel_num; voxel_idx++) {
    if (!voxels_validity[voxel_idx] || is_processed[voxel_idx]) {
      is_processed[voxel_idx] = true;
      continue;
    }

    std::vector<bool> visited(voxel_num, false);
    std::vector<uint> segment;
    segment.reserve(50);
    std::deque<uint> queue;
    queue.push_back(voxel_idx);

    while(!queue.empty()) {
      // Dequeue a segment
      uint supervoxel_idx = queue.front();
      queue.pop_front();

      segment.push_back(supervoxel_idx);

      for (auto neigh_idx : supervoxels_adjacency[supervoxel_idx]) {
        if (!voxels_validity[neigh_idx] || visited[neigh_idx])
          continue;

        is_processed[neigh_idx] = true;
        visited[neigh_idx] = true;

        bool same_proj = true;
        for (uint seg_elt : segment)
          // same_proj = same_proj && (supervoxels_normals[supervoxel_idx].dot(supervoxels_normals[seg_elt]) > 0.f);
          same_proj = same_proj && (supervoxels_normals[supervoxel_idx].dot(supervoxels_normals[seg_elt]) > cosf(M_PI*5.f/9.f));

        if (!same_proj)
          continue;

        convexity_conf = convexityConfidence(supervoxels_centers[supervoxel_idx],
                                             supervoxels_centers[neigh_idx],
                                             supervoxels_normals[supervoxel_idx],
                                             supervoxels_normals[neigh_idx]);

        // If the connection is convex, segments can be flat or convex to be merged
        if (convexity_conf >= 0.48 && convexity_bels[voxel_idx] >= 0.45 && convexity_bels[neigh_idx] >= 0.45)
          queue.push_back(neigh_idx);

        // If the connection is concave, segments have to be concave to be merged
        if (convexity_conf < 0.48 && convexity_bels[voxel_idx] < 0.45 && convexity_bels[neigh_idx] < 0.45)
          queue.push_back(neigh_idx);

      }
    }


    // Overlap
    std::unordered_map<uint, std::vector<uint> > overlapping_segments;

    for (auto supervoxel_idx : segment)
      for (auto seg_idx : supervoxels_to_segments[supervoxel_idx])
        overlapping_segments[seg_idx].push_back(supervoxel_idx);

    std::cout << "-------------------------------------------" << std::endl;
    for (auto p : overlapping_segments) {
      std::cout << 100.f * static_cast<float>(p.second.size()) / static_cast<float>(segment.size()) << "% with " << p.first << std::endl;
    }

    // segments_to_supervoxels.push_back(segment);
    for (auto supervoxel_idx : segment)
      supervoxels_to_segments[supervoxel_idx].push_back(segments_to_supervoxels.size());
    segments_to_supervoxels.push_back(std::move(segment));

  }
} // -- end ScopeTime t("Segment Merging", debug_);


{
  ScopeTime t("Extract clusters info", debug_);

  for (uint seg_idx=0; seg_idx<segments_to_supervoxels.size(); seg_idx++) {
    Eigen::Vector3f segment_mean = Eigen::Vector3f::Zero();
    for (auto supervoxel_idx : segments_to_supervoxels[seg_idx])
      segment_mean += supervoxels_centers[supervoxel_idx] / static_cast<float>(segments_to_supervoxels[seg_idx].size());

    float min_centroid_dist = 1e5f;
    uint centroid_idx;
    for (auto supervoxel_idx : segments_to_supervoxels[seg_idx]) {
      float centroid_dist = (segment_mean - supervoxels_centers[supervoxel_idx]).norm();

      if (centroid_dist < min_centroid_dist) {
        min_centroid_dist = centroid_dist;
        centroid_idx = supervoxel_idx;
      }
    }

    support_points.push_back((*(supervoxels_indices[centroid_idx]))[0]);
  }


  for (uint seg_idx=0; seg_idx<segments_to_supervoxels.size(); seg_idx++) {
    std::vector<std::pair<uint, int> > region_grown;
    region_grown.reserve(100);

    pcl::IndicesPtr indices(new std::vector<int>());
    indices->reserve(100);

    for (auto supervoxel_idx : segments_to_supervoxels[seg_idx]) {
      indices->insert(indices->end(),
                     (*(supervoxels_indices[supervoxel_idx])).begin(),
                     (*(supervoxels_indices[supervoxel_idx])).end());

      for (auto pt_idx : *(supervoxels_indices[supervoxel_idx])) {
        auto pair = std::make_pair(pt_idx, 0);
        region_grown.push_back(pair);
      }
    }

    if (region_grown.size() < 3)
      continue;

    std::cout << "Segment " << seg_idx << " nb supervoxels " << segments_to_supervoxels[seg_idx].size() << " np points " << region_grown.size() << std::endl;

    // --- Get LRF from the region of interest -------------------------------------------------------------------
    Eigen::Matrix3f lrf;
    Eigen::Vector3f mean;
    normalAlignedPca(indices, lrf, mean);
    lrf_transforms.push_back(lrf);
    means.push_back(mean);
    scales.push_back(regionScale(indices, mean));
    support_regions.push_back(std::move(*indices));
  }


  neighbor_indices.resize(support_points.size());
  for (uint seg_idx=0; seg_idx<segments_to_supervoxels.size(); seg_idx++) {

    std::unordered_set<uint> neighboring_supervoxels;
    for (auto supervoxel_idx : segments_to_supervoxels[seg_idx])
      for (auto neigh_supervoxel_idx : supervoxels_adjacency[supervoxel_idx])
        neighboring_supervoxels.insert(neigh_supervoxel_idx);

    std::unordered_set<uint> neighboring_segments;
    for (auto neigh_supervoxel_idx : neighboring_supervoxels)
        for (auto neigh_segment_idx : supervoxels_to_segments[neigh_supervoxel_idx])
          neighboring_segments.insert(neigh_segment_idx);

    for (auto seg : neighboring_segments)
      neighbor_indices[seg_idx].push_back(seg);
  }

} // -- end ScopeTime t("Extract clusters info", debug_);





  // --- Visualization -------------------------------------------------------------------------------------------
  if (debug_)
    std::cout << "Seed: " << seed << std::endl;

  srand(seed);



  // // for (auto& rgb : voxel_colors) {
  // for (uint voxel_idx=0; voxel_idx < voxel_num; voxel_idx++) {
  //   auto& rgb = voxel_colors[voxel_idx];
  //   std::get<0>(rgb) = rand() % 256;
  //   std::get<1>(rgb) = rand() % 256;
  //   std::get<2>(rgb) = rand() % 256;

  //   // if (convexity_bels[voxel_idx] > 0.6) {
  //   //   std::get<0>(rgb) = 0;
  //   //   std::get<1>(rgb) = 255;
  //   //   std::get<2>(rgb) = 0;
  //   // } else if (convexity_bels[voxel_idx] < 0.4) {
  //   //   std::get<0>(rgb) = 255;
  //   //   std::get<1>(rgb) = 0;
  //   //   std::get<2>(rgb) = 0;
  //   // } else {
  //   //   std::get<0>(rgb) = 255;
  //   //   std::get<1>(rgb) = 255;
  //   //   std::get<2>(rgb) = 255;
  //   // }
  // }

  // for (uint pt_idx=0; pt_idx<pc_->points.size(); pt_idx++) {

  //   // if (pt_supervoxel_indices[pt_idx] == 1e5)
  //   if (pt_merged_voxel_indices[pt_idx] == 1e5)
  //     continue;
  //   pcl::PointXYZRGB p;
  //   p.x = pc_->points[pt_idx].x;
  //   p.y = pc_->points[pt_idx].y;
  //   p.z = pc_->points[pt_idx].z;

  //   // if (pt_supervoxel_indices[pt_idx] == 1e5) {
  //   //   p.r = 255;
  //   //   p.g = 0;
  //   //   p.b = 0;
  //   // } else {
  //   //   p.r = 255;
  //   //   p.g = 255;
  //   //   p.b = 255;
  //   // }
  //   // p.r = std::get<0>(voxel_colors[pt_supervoxel_indices[pt_idx]]);
  //   // p.g = std::get<1>(voxel_colors[pt_supervoxel_indices[pt_idx]]);
  //   // p.b = std::get<2>(voxel_colors[pt_supervoxel_indices[pt_idx]]);

  //   p.r = std::get<0>(voxel_colors[pt_merged_voxel_indices[pt_idx]]);
  //   p.g = std::get<1>(voxel_colors[pt_merged_voxel_indices[pt_idx]]);
  //   p.b = std::get<2>(voxel_colors[pt_merged_voxel_indices[pt_idx]]);

  //   cloud->points.push_back(p);
  // }

  // for (uint voxel_idx=0; voxel_idx<voxel_num; voxel_idx++) {
  //   if (!voxels_validity[voxel_idx])
  //     continue;

  //   PointT p1 = voxels_centers[voxel_idx];

  //   for (auto neigh_idx : supervoxels_adjacency[voxel_idx]) {
  //     PointT p2 = voxels_centers[neigh_idx];
  //     viewer->addLine<PointT>(p1, p2, 0., 0., 1., "line_" +std::to_string(voxel_idx)+"_"+std::to_string(neigh_idx));
  //     // std::cout << "line_" +std::to_string(voxel_idx)+"_"+std::to_string(neigh_idx) << std::endl;
  //   }
  // }

  // viewer->addPointCloud<PointT> (pc_, "pc_");
  // viewer->addPointCloud<pcl::PointXYZRGB> (cloud, "voxels");
  // while (!viewer->wasStopped()) {
  //   viewer->spinOnce(100);
  // }


  if (debug_)
    std::cout << "Sampled " << support_points.size() << " supports points" << std::endl;

  return 0;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// std::pair<Eigen::Matrix3f, Eigen::Vector3f>
void  GraphConstructor::normalAlignedPca(const pcl::IndicesPtr & indices,
                                         Eigen::Matrix3f & lrf,
                                         Eigen::Vector3f & mean) {

  pcl::PCA<PointT> pca;
  pca.setInputCloud(pc_);
  pca.setIndices(indices);

  // Column ordered eigenvectors, representing the eigenspace cartesian basis (right-handed coordinate system).
  lrf = pca.getEigenVectors().transpose();
  mean = pca.getMean().head<3>();

  // // "EYELRF"  : 1
  // if (lrf_code_ == 1)
  //   return std::make_pair(Eigen::Matrix3f::Identity(), mean);

  // Align with normals
  uint sampled_nb = std::min<uint>(static_cast<uint>(100), indices->size());
  uint plusNormals = 0, pt_idx;

  for (uint i=0; i<sampled_nb; i++) {
    pt_idx = (*indices)[i];
    Eigen::Vector3f n = pc_->points[pt_idx].getNormalVector3fMap();

    if (n.dot(lrf.row(2)) > 0.)
      plusNormals++;
  }

  // If less than half aligns, flip the LRF
  if (2*plusNormals < sampled_nb)
    lrf.row(2) = -lrf.row(2);

  // Update the properties of the graph
  lrf.row(1).matrix () = lrf.row(2).cross (lrf.row(0));

  // // "PCALRF"  : 2
  // if (lrf_code_ == 2)
  //   return std::make_pair(lrf, mean);

  // Eigen::Vector3f z(0.f, 0.f, 1.f);
  // Eigen::Matrix3f zlrf = Eigen::Matrix3f::Identity();
  // Eigen::Vector3f node_normal = lrf.row(2);
  // node_normal(2) = 0.f;
  // node_normal.normalize();
  // if (std::isnan(node_normal(0)))
  //   node_normal << 1., 0., 0.;
  // zlrf.row(0) = node_normal;
  // zlrf.row(1) = z.cross(node_normal);
  // zlrf.row(2) = z;


  // // "ZLRF"    : 3
  // if (lrf_code_ == 3)
  //   return std::make_pair(zlrf, mean);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float GraphConstructor::regionScale(const pcl::IndicesPtr & indices,
                                    Eigen::Vector3f & mean) {
  float max_dist = 0.f, d;

  for (auto pt : *indices) {
    d = (pc_->points[pt].getVector3fMap() - mean).norm();

    if (d > max_dist)
      max_dist = d;
  }

  return 1. / max_dist;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::searchSupportPointsNeighbors(std::vector<uint> & support_points,
                                                    std::vector<std::vector<uint> > & neighbor_indices,
                                                    int* neigh_indices, uint neighbors_nb,
                                                    float shadowing_threshold, int* valid_indices) {

  float shadowing_threshold_rad = shadowing_threshold * M_PI / 180.;
  Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > neigh_indices_map(neigh_indices, support_points.capacity(), neighbors_nb);

  // std::unordered_map<uint, uint> pc_to_support_indices;

  // for (uint support_pt_idx=0; support_pt_idx<support_points.size(); support_pt_idx++) {
  //   pc_to_support_indices[support_points[support_pt_idx]] = support_pt_idx;
  // }

  uint pt_idx, max_neigh;
  for (uint support_pt_idx=0; support_pt_idx<support_points.size(); support_pt_idx++) {
    if (valid_indices[support_pt_idx] != 1)
        continue;

    pt_idx = support_points[support_pt_idx];
    max_neigh = std::min<uint>(neighbor_indices[support_pt_idx].size()+1, neighbors_nb);

    neigh_indices_map(support_pt_idx, 0) = support_pt_idx;

    uint neigh_idx = 0;
    Eigen::Vector3f support_vector = pc_->points[support_points[support_pt_idx]].getVector3fMap();
    Eigen::Vector3f support_normal = pc_->points[support_points[support_pt_idx]].getNormalVector3fMap();

    for (uint i=0; i<max_neigh-1; i++) {
      // uint neigh_pt_idx = pc_to_support_indices[neighbor_indices[pt_idx][i]];
      uint neigh_pt_idx = neighbor_indices[support_pt_idx][i];
      if (valid_indices[neigh_pt_idx] != 1)
        continue;

      Eigen::Vector3f neigh_vector = pc_->points[support_points[neigh_pt_idx]].getVector3fMap() - support_vector;
      neigh_vector.normalize();

      // Is it shadowed by the opposite of the support point normal
      bool shadowed = false;
      float angle = acosf(std::max(-1.0f, std::min(1.0f, (neigh_vector.dot(-support_normal)))));
      if (angle < shadowing_threshold_rad)
        shadowed = true;

      // Is it shadowed by another neighbor
      for (uint j=0; j<neigh_idx; j++) {
        Eigen::Vector3f prev_neigh_vector = pc_->points[support_points[neighbor_indices[support_pt_idx][j]]].getVector3fMap() - support_vector;
        prev_neigh_vector.normalize();
        angle = acosf(std::max(-1.0f, std::min(1.0f, (neigh_vector.dot(prev_neigh_vector)))));
        if (angle < shadowing_threshold_rad)
          shadowed = true;
      }

      if (!shadowed) {
        // neigh_indices_map(support_pt_idx, neigh_idx+1) = pc_to_support_indices[neighbor_indices[pt_idx][i]];
        neigh_indices_map(support_pt_idx, neigh_idx+1) = neigh_pt_idx;
        neigh_idx++;
      }
    }

  }

  // TODO Should I BFS through the graph to get some more neighbors ?
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////// VIZ //////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::vizGraph(int max_support_point, float neigh_size, uint neighbors_nb) {
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  boost::shared_ptr<VizData> user_data(new VizData());

  user_data->graph = this;
  user_data->viewer = viewer.get();
  user_data->max_support_point = max_support_point;
  user_data->neigh_size = neigh_size;
  user_data->neighbors_nb = neighbors_nb;

  viewer->registerKeyboardCallback(keyboardEventOccurred, (void*)user_data.get ());
  // viewer->setBackgroundColor(1., 1., 1.);
  viewer->setBackgroundColor(0., 0., 0.);

  // pcl::PointCloud<Poin>
  // for (uint i=0; i<pc_->points.size(); i++) {
  //   pc_->points[i].intensity = 1.0;
  // }

  // pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(pc_);
    // viewer->addPointCloud<pcl::PointXYZRGB> (support_pc, rgb, "vertices");

  viewer->addPointCloud<PointT> (pc_, "pc_");


  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
  }
}

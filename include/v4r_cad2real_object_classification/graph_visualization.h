#pragma once
#include <pcl/visualization/pcl_visualizer.h>


struct VizData {
  GraphConstructor * graph;
  pcl::visualization::PCLVisualizer * viewer;
  bool pc=true;
  bool normals=false;
  bool w_edges=false;
  bool w_lrf=false;
  bool lrf=false;
  bool normal_occlusion=false;
  int max_support_point=32;
  uint neighbors_nb=3;
  uint prev_support_pts_nb=0;
};


void getJetColour(float v,
                  const float vmin,
                  const float vmax,
                  pcl::PointXYZRGB & p)
{
   p.r = p.g = p.b = 255;
   float dv;

   if (v < vmin)
      v = vmin;
   if (v > vmax)
      v = vmax;
   dv = vmax - vmin;

   if (v < (vmin + 0.25 * dv)) {
      p.r = 0;
      p.g = static_cast<int>(255.*(4 * (v - vmin) / dv));
   } else if (v < (vmin + 0.5 * dv)) {
      p.r = 0;
      p.b = static_cast<int>(255.*(1 + 4 * (vmin + 0.25 * dv - v) / dv));
   } else if (v < (vmin + 0.75 * dv)) {
      p.r = static_cast<int>(255.*(4 * (v - vmin - 0.5 * dv) / dv));
      p.b = 0;
   } else {
      p.g = static_cast<int>(255.*(1 + 4 * (vmin + 0.75 * dv - v) / dv));
      p.b = 0;
   }
}


/*
 * H(Hue): 0 - 360 degree (integer)
 * S(Saturation): 0 - 1.00 (double)
 * V(Value): 0 - 1.00 (double)
 *
 * output[3]: Output, array size 3, int
 */
void HSVtoRGB(int H, double S, double V,
              pcl::PointXYZRGB & p) {
  double C = S * V;
  double X = C * (1 - fabs(fmod(H / 60.0, 2) - 1.));
  double m = V - C;
  double Rs, Gs, Bs;

  if(H >= 0 && H < 60) {
    Rs = C;
    Gs = X;
    Bs = 0;
  }
  else if(H >= 60 && H < 120) {
    Rs = X;
    Gs = C;
    Bs = 0;
  }
  else if(H >= 120 && H < 180) {
    Rs = 0;
    Gs = C;
    Bs = X;
  }
  else if(H >= 180 && H < 240) {
    Rs = 0;
    Gs = X;
    Bs = C;
  }
  else if(H >= 240 && H < 300) {
    Rs = X;
    Gs = 0;
    Bs = C;
  }
  else {
    Rs = C;
    Gs = 0;
    Bs = X;
  }

  p.r = static_cast<int>((Rs + m) * 255);
  p.g = static_cast<int>((Gs + m) * 255);
  p.b = static_cast<int>((Bs + m) * 255);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event,
                            void* user_data_void) {
  /* PREDEFINED KEYBINDINGS
          p, P   : switch to a point-based representation
          w, W   : switch to a wireframe-based representation (where available)
          s, S   : switch to a surface-based representation (where available)

          j, J   : take a .PNG snapshot of the current window view
          c, C   : display current camera/window parameters
          f, F   : fly to point mode

          e, E   : exit the interactor
          q, Q   : stop and call VTK's TerminateApp

           +/-   : increment/decrement overall point size
     +/- [+ ALT] : zoom in/out

          g, G   : display scale grid (on/off)
          u, U   : display lookup table (on/off)

    r, R [+ ALT] : reset camera [to viewpoint = {0, 0, 0} -> center_{x, y, z}]

    ALT + s, S   : turn stereo mode on/off
    ALT + f, F   : switch between maximized window mode and original size

          l, L   : list all available geometric and color handlers for the current actor map
    ALT + 0..9 [+ CTRL]  : switch between different geometric handlers (where available)
          0..9 [+ CTRL]  : switch between different color handlers (where available)

    SHIFT + left click   : select a point
*/

  VizData *user_data = static_cast<VizData *> (user_data_void);
  pcl::visualization::PCLVisualizer *viewer = user_data->viewer;
  pcl::PointCloud<PointT>::Ptr pc_ = (user_data->graph)->pc_;

  if ((event.getKeySym () == "v") &&
      event.keyDown ()) {

    viewer->removePointCloud("vertices");
    viewer->removePointCloud("coverage");
    viewer->removePointCloud("centroids");
    viewer->removePointCloud("colors");
    viewer->removeAllShapes();
    if (user_data->w_lrf) {
      for (uint i=0; i<user_data->prev_support_pts_nb; i++)
        viewer->removeCoordinateSystem("lrf_"+std::to_string(i));
    }

    if (event.isAltPressed()) {
      user_data->w_lrf = !user_data->w_lrf;
      std::cout << "ALT+v was pressed => display LRF" << std::endl;
    }

    if (event.isCtrlPressed()) {
      user_data->w_edges = !user_data->w_edges;
      std::cout << "CTRL+v was pressed => display vertices w/ edges" << std::endl;
    }

    struct timeval time;
    gettimeofday(&time,NULL);
    int seed = (time.tv_sec * 1000) + (time.tv_usec / 1000);

    std::vector<std::vector<uint> > neighbor_indices;
    std::vector<Eigen::Matrix3f> lrf_transforms_vec;
    std::vector<float> scales;
    std::vector<Eigen::Vector3f> means;
    std::vector<std::vector<int> > support_regions;
    (user_data->graph)->sampleSegments(neighbor_indices, lrf_transforms_vec, scales, means, support_regions,
                                       user_data->max_support_point, user_data->neighbors_nb, seed);

    std::cout << "support_points: " << means.size() << " neighbor_indices: " << neighbor_indices.size() << " lrf_transforms_vec: " << lrf_transforms_vec.size() << std::endl;
    user_data->prev_support_pts_nb = means.size();


    pcl::PointCloud<pcl::PointXYZRGB>::Ptr coverage(new pcl::PointCloud<pcl::PointXYZRGB>);
    coverage->points.resize(pc_->points.size());
    std::vector<bool> ignored(pc_->points.size(), true);

    for (auto region : support_regions) {
      for (auto pt_idx : region)
        ignored[pt_idx] = false;
    }

    for (uint pt_idx=0; pt_idx<pc_->points.size(); pt_idx++) {
      pcl::PointXYZRGB p;
      p.x = pc_->points[pt_idx].x;
      p.y = pc_->points[pt_idx].y;
      p.z = pc_->points[pt_idx].z;

      if (ignored[pt_idx]) {
        p.r = 255;
        p.g = 240;
        p.b = 240;
      } else {
        p.r = 40;
        p.g = 40;
        p.b = 40;
      }

      coverage->points[pt_idx] = p;
    }



    pcl::PointCloud<pcl::PointXYZRGB>::Ptr support_pc(new pcl::PointCloud<pcl::PointXYZRGB>);

    support_pc->points.resize(means.size());
    for (uint i=0; i<means.size(); i++) {
      pcl::PointXYZRGB p;
      p.x = means[i](0);
      p.y = means[i](1);
      p.z = means[i](2);

      int H = static_cast<int>(static_cast<float>(i) / means.size() * 360);
      // HSVtoRGB(H, 1.0, 1.0, p);

      p.r = static_cast<int> (rand() / (static_cast<double> (RAND_MAX/255)));
      p.g = static_cast<int> (rand() / (static_cast<double> (RAND_MAX/255)));
      p.b = static_cast<int> (rand() / (static_cast<double> (RAND_MAX/255)));

      support_pc->points[i] = p;
    }

    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(support_pc);
    viewer->addPointCloud<pcl::PointXYZRGB> (support_pc, rgb, "vertices");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "vertices");

    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_coverage(coverage);
    viewer->addPointCloud<pcl::PointXYZRGB> (coverage, rgb_coverage, "coverage");


    // --- Add colored point cloud (based on grown regions) ------------------------------------------------------
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colors_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
    colors_pc->points.reserve(pc_->points.size());

    for (uint seg_idx=0; seg_idx<support_regions.size(); seg_idx++) {
      int r = support_pc->points[seg_idx].r;
      int g = support_pc->points[seg_idx].g;
      int b = support_pc->points[seg_idx].b;
      // Eigen::Vector3f offset = lrf_transforms_vec[seg_idx].row(2);
      Eigen::Vector3f offset = means[seg_idx];
      offset.normalize();
      // offset*= .15f;
      offset*= 1.1f;

      for (auto pt_idx : support_regions[seg_idx]) {
        pcl::PointXYZRGB p;
        p.x = pc_->points[pt_idx].x; // + offset(0);
        p.y = pc_->points[pt_idx].y; // + offset(1);
        p.z = pc_->points[pt_idx].z; // + offset(2);
        p.r = r;
        p.g = g;
        p.b = b;

        colors_pc->points.push_back(p);
      }
    }

    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_colors(colors_pc);
    viewer->addPointCloud<pcl::PointXYZRGB> (colors_pc, rgb_colors, "colors");


    // --- Add edges between connected points --------------------------------------------------------------------
    if (user_data->w_edges) {
      for (uint seg_idx=0; seg_idx<means.size(); seg_idx++) {
        pcl::PointXYZRGB p1 = support_pc->points[seg_idx];
        uint max_neigh = std::min<uint>(neighbor_indices[seg_idx].size(), user_data->neighbors_nb);

        for (uint col_idx=0; col_idx<max_neigh; col_idx++) {
          pcl::PointXYZRGB p2;
          p2.x = means[neighbor_indices[seg_idx][col_idx]](0);
          p2.y = means[neighbor_indices[seg_idx][col_idx]](1);
          p2.z = means[neighbor_indices[seg_idx][col_idx]](2);

          viewer->addLine<pcl::PointXYZRGB>(p1, p2, 0., 0., 1.,
                                            "line_" +std::to_string(seg_idx)+"_"+std::to_string(neighbor_indices[seg_idx][col_idx]));
        }
      }
    }


    // --- Add LRF to support points -----------------------------------------------------------------------------
    if (user_data->w_lrf) {
      for (uint seg_idx=0; seg_idx< lrf_transforms_vec.size(); seg_idx++) {
        Eigen::Vector4f v;
        v << means[seg_idx](0), means[seg_idx](1), means[seg_idx](2), 1.f;
        float norm1 = 5. * scales[seg_idx];

        if (std::isinf(norm1))
          std::cout << lrf_transforms_vec[seg_idx] << std::endl;

        Eigen::Matrix4f Trans;
        Trans.setIdentity();
        Trans.block<3,3>(0,0) = lrf_transforms_vec[seg_idx].transpose();
        Trans.rightCols<1>() = v;

        Eigen::Affine3f F;
        F = Trans;
        viewer->addCoordinateSystem(1 / norm1, F, "lrf_"+std::to_string(seg_idx));
      }
    }

  } else if (event.getKeySym () == "m" &&
             event.keyDown ()) {
    user_data->normal_occlusion = !user_data->normal_occlusion;
    std::cout << "m was pressed => normal occlusion" << std::endl;

  } else if (event.getKeySym () == "n" &&
             event.keyDown ()) {
    user_data->normals = !user_data->normals;
    std::cout << "n was pressed => display normals" << std::endl;
    if (user_data->normals)
      viewer->addPointCloudNormals<PointT, PointT> (pc_, pc_, 10, 0.02, "normals");
    else
      viewer->removePointCloud("normals");

  } else if (event.getKeySym () == "p" &&
             event.keyDown ()) {
    std::cout << "p was pressed => display pc_" << std::endl;
    user_data->pc = !user_data->pc;
    if (user_data->pc)
      viewer->addPointCloud<PointT> (pc_, "pc_");
    else
      viewer->removePointCloud("pc_");

  }
}

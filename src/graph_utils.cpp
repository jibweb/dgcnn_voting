#include <fstream>
#include <vector>

#include <pcl/common/projection_matrix.h>
#include <tinyply.h>


union Int {
      int i;
      char s[sizeof(int)];
};


namespace tinyply {
  union Float
  {
      float f;
      char s[sizeof(float)];
  };

  union Uint
  {
      uint32_t f;
      char s[sizeof(uint32_t)];
  };

  template<class PointT>
  void loadPLY(const std::string filepath,
               pcl::PointCloud<PointT> & pc,
               std::vector<std::array<uint32_t, 3> > & triangles) {

    std::ifstream ss(filepath, std::ios::binary);
    if (ss.fail()) throw std::runtime_error("failed to open " + filepath);

    PlyFile file;
    file.parse_header(ss);

    // Tinyply treats parsed data as untyped byte buffers. See below for examples.
    std::shared_ptr<PlyData> ply_verts, ply_norms, ply_faces;

    // The header information can be used to programmatically extract properties on elements
    // known to exist in the header prior to reading the data. For brevity of this sample, properties
    // like vertex position are hard-coded:
    try { ply_verts = file.request_properties_from_element("vertex", { "x", "y", "z" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    try { ply_norms = file.request_properties_from_element("vertex", { "nx", "ny", "nz" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    // Providing a list size hint (the last argument) is a 2x performance improvement. If you have
    // arbitrary ply files, it is best to leave this 0.
    try { ply_faces = file.request_properties_from_element("face", { "vertex_indices" }, 3); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }


    file.read(ss);

    // Copying the buffer into the point cloud (both coordinates and normals)
    uint8_t * vert_buff_ptr = ply_verts->buffer.get();
    uint8_t * norm_buff_ptr = ply_norms->buffer.get();
    pc.points.resize(ply_verts->count);
    for (uint pt_idx=0; pt_idx<ply_verts->count; pt_idx++) {
      for (uint i=0; i<3; i++) {
        Float vert_f, norm_f;
        vert_f.s[0] = *vert_buff_ptr++;
        vert_f.s[1] = *vert_buff_ptr++;
        vert_f.s[2] = *vert_buff_ptr++;
        vert_f.s[3] = *vert_buff_ptr++;

        norm_f.s[0] = *norm_buff_ptr++;
        norm_f.s[1] = *norm_buff_ptr++;
        norm_f.s[2] = *norm_buff_ptr++;
        norm_f.s[3] = *norm_buff_ptr++;

        pc.points[pt_idx].data[i] = vert_f.f;
        pc.points[pt_idx].normal[i] = norm_f.f;
      }
    }

    // Copying the triangles in the pcl::PoylgonMesh
    uint8_t * face_buff_ptr = ply_faces->buffer.get();
    // mesh.polygons.resize(ply_faces->count);
    triangles.resize(ply_faces->count);
    for (uint tri_idx=0; tri_idx<ply_faces->count; tri_idx++) {
      for (uint i=0; i<3; i++) {
        Uint f;
        f.s[0] = *face_buff_ptr++;
        f.s[1] = *face_buff_ptr++;
        f.s[2] = *face_buff_ptr++;
        f.s[3] = *face_buff_ptr++;

        triangles[tri_idx][i] = f.f;
      }
    }


    //TODO COMPUTE NORMALS IF NOT FOUND

  }
}


inline double beta_22(double x) {
  return (-x*x*x/3 + x*x/2) / (-1./3. + 1./2.);
}


inline double beta_23(double x) {
  return (x*x*x*x/4. - 2.*x*x*x/3. + x*x/2.) / (1./4. - 2./3. + 1./2.);
}


std::vector<double> compute_factors(int max_depth, double func (double)) {
  std::vector<double> factors(max_depth+1, 0.);

  for(uint i=0; i<factors.size(); i++) {
    double x = static_cast<double> (i) / static_cast<double> (max_depth);
    factors[i] = func(x);
  }

  return factors;
}


float triangle_area(Eigen::Vector4f& p1, Eigen::Vector4f& p2, Eigen::Vector4f& p3) {
  float a,b,c,s;

  // Get the area of the triangle
  Eigen::Vector4f v21 (p2 - p1);
  Eigen::Vector4f v31 (p3 - p1);
  Eigen::Vector4f v23 (p2 - p3);
  a = v21.norm (); b = v31.norm (); c = v23.norm (); s = (a+b+c) * 0.5f + 1e-6;

  return sqrt(s * (s-a) * (s-b) * (s-c));
}


Eigen::Vector3f triangle_normal(Eigen::Vector3f& p1, Eigen::Vector3f& p2, Eigen::Vector3f& p3) {

  // Get the area of the triangle
  Eigen::Vector3f v21 (p2 - p1);
  Eigen::Vector3f v31 (p3 - p1);

  Eigen::Vector3f normal = v21.cross(v31);
  normal.normalize();

  return normal;
}

inline double convexityConfidence(Eigen::Vector3f x1, Eigen::Vector3f x2,
                           Eigen::Vector3f n1, Eigen::Vector3f n2) {
    Eigen::Vector3f d = x1 - x2;
    d.normalize();

    // The closer to -2, the more convex
    double local_convexity_measure = static_cast<double>(n1.dot(d)) - static_cast<double>(n2.dot(d));
    local_convexity_measure = (local_convexity_measure + 2.)/4.;
    return std::min(std::max(0.0001, local_convexity_measure), 0.9999);
}

#cython: embedsignature=True
#cython: language_level=3

from __future__ import print_function

import cython
import datetime
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdlib cimport malloc, free
# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np


# Declare the class with cdef
cdef extern from "v4r_cad2real_object_classification/py_graph_construction.h":
    # ctypedef struct TransfoParams:
    #     # PC transformations
    #     float to_remove
    #     unsigned int to_keep
    #     float occl_pct
    #     float noise_std
    #     unsigned int rotation_deg
    cdef cppclass GraphConstructor:
        GraphConstructor(bool debug, int lrf_code, bool wnormals) except +

        # General
        void initializePLY(string filename, bool height_to_zero)
        void initializePCD(string filename, string filename_adj)
        void initializeScanObjectNN(string filename, string filename_adj, int filter_class_idx, bool height_to_zero)
        void initializeArray(float* vertices, unsigned int vertex_nb, int* triangles, unsigned int triangle_nb)
        void initializeArray(float* vertices, unsigned int vertex_nb, int* triangles, unsigned int triangle_nb, float* normals)
        void dataAugmentation(bool normal_smoothing, float normal_occlusion, float normal_noise)
        void getBbox(float* bbox)
        int pySampleSegments(float* support_points_coords, int* neigh_indices, float* lrf_transforms,
                             int* valid_indices, float* scales, int max_support_point, float neigh_size,
                             int neighbors_nb, float shadowing_threshold, int seed, float** regions, unsigned int region_sample_size,
                             int disconnect_rate, float* heights)
        void vizGraph(int max_support_point, float neigh_size, unsigned int neighbors_nb)


class NanNormals(Exception):
    """Nan found in the normal of the model"""
    pass


cdef class PyGraph:
    cdef GraphConstructor*c_graph  # Hold a C++ instance which we're wrapping
    cdef bool wnormals

    def __cinit__(self, debug=True, lrf=2, wnormals=False):
        self.c_graph = new GraphConstructor(debug, lrf, wnormals)
        self.wnormals = wnormals

    def initialize_from_ply(self, fn, height_to_zero=False):
        self.c_graph.initializePLY(fn, height_to_zero)

    def initialize_from_pcd(self, string fn, string adj_fn):
        self.c_graph.initializePCD(fn, adj_fn)

    def initialize_from_scanobjectnn(self, string fn, string adj_fn, int filter_class_idx=-1, height_to_zero=False):
        self.c_graph.initializeScanObjectNN(fn, adj_fn, filter_class_idx, height_to_zero)

    def initialize_from_array(self, np.ndarray[float, ndim=2, mode="c"] vertices,
                              np.ndarray[int, ndim=2, mode="c"] triangles,
                              np.ndarray[float, ndim=2, mode="c"] normals=None):
        if normals:
            self.c_graph.initializeArray(&vertices[0, 0], vertices.shape[0],
                                         &triangles[0, 0], triangles.shape[0],
                                         &normals[0, 0])
        else:
            self.c_graph.initializeArray(&vertices[0, 0], vertices.shape[0],
                                         &triangles[0, 0], triangles.shape[0])

    def data_augmentation(self,
                          normal_smoothing=False,
                          float normal_occlusion=-2.,
                          float normal_noise=0.):
        self.c_graph.dataAugmentation(normal_smoothing, normal_occlusion, normal_noise)

    # def sample_support_points(self, float neigh_size,
    #                           int max_support_point=1024,
    #                           int neighbors_nb=3,
    #                           float shadowing_threshold=10.,
    #                           int seed=-1):
    #     cdef np.ndarray[float, ndim=2, mode="c"] support_points = \
    #         np.zeros([max_support_point, 3], dtype=np.float32)

    #     cdef np.ndarray[int, ndim=2, mode="c"] neigh_indices = \
    #         - np.ones([max_support_point, neighbors_nb], dtype=np.int32)

    #     cdef np.ndarray[float, ndim=2, mode="c"] lrf_transforms = \
    #         np.zeros([max_support_point, 9], dtype=np.float32)

    #     cdef np.ndarray[int, ndim=1, mode="c"] valid_indices = \
    #         np.zeros([max_support_point], dtype=np.int32)

    #     cdef np.ndarray[float, ndim=1, mode="c"] scales = \
    #         np.zeros([max_support_point], dtype=np.float32)

    #     cdef np.ndarray[float, ndim=2, mode="c"] normals = \
    #         np.zeros([max_support_point, 3], dtype=np.float32)

    #     cdef cseed
    #     if seed != -1:
    #         cseed = <int> seed
    #     else:
    #         now = datetime.datetime.now()
    #         cseed = <int> (1000*now.second + now.microsecond/100)

    #     result = self.c_graph.sampleSupportPoints(&support_points[0, 0],
    #                                               &neigh_indices[0, 0],
    #                                               &lrf_transforms[0, 0],
    #                                               &valid_indices[0],
    #                                               &scales[0],
    #                                               max_support_point,
    #                                               neigh_size,
    #                                               neighbors_nb,
    #                                               shadowing_threshold,
    #                                               cseed,
    #                                               &normals[0, 0])

    #     if result == -1:
    #         raise NanNormals
    #     else:
    #         if self.wnormals:
    #             return support_points, neigh_indices, \
    #                 lrf_transforms.reshape((-1, 3, 3)), valid_indices, \
    #                 normals
    #         else:
    #             return support_points, neigh_indices, \
    #                 lrf_transforms.reshape((-1, 3, 3)), valid_indices, scales

    def sample_support_points_and_regions(self, float neigh_size,
                                          int max_support_point=1024,
                                          int neighbors_nb=3,
                                          float shadowing_threshold=10.,
                                          int region_sample_size=64,
                                          int disconnect_rate=0,
                                          int seed=-1):
        cdef np.ndarray[float, ndim=2, mode="c"] support_points = \
            np.zeros([max_support_point, 3], dtype=np.float32)

        cdef np.ndarray[int, ndim=2, mode="c"] neigh_indices = \
            - np.ones([max_support_point, neighbors_nb], dtype=np.int32)

        cdef np.ndarray[float, ndim=2, mode="c"] lrf_transforms = \
            np.zeros([max_support_point, 9], dtype=np.float32)

        cdef np.ndarray[int, ndim=1, mode="c"] valid_indices = \
            np.zeros([max_support_point], dtype=np.int32)

        cdef np.ndarray[float, ndim=1, mode="c"] scales = \
            np.zeros([max_support_point], dtype=np.float32)

        cdef np.ndarray[float, ndim=1, mode="c"] heights = \
            np.zeros([max_support_point], dtype=np.float32)

        cdef np.ndarray[float, ndim=1, mode="c"] bbox = \
            np.zeros([6], dtype=np.float32)

        self.c_graph.getBbox(&bbox[0])

        cdef float **feats2d_ptr = <float **> malloc((max_support_point)*sizeof(float *))
        feats2d = []
        cdef np.ndarray[float, ndim=2, mode="c"] tmp
        if self.wnormals:
            arr_shape = [region_sample_size, 6]
        else:
            arr_shape = [region_sample_size, 3]

        for i in range(max_support_point):
            tmp = np.zeros(arr_shape, dtype=np.float32)
            feats2d_ptr[i] = &tmp[0, 0]
            feats2d.append(tmp)

        cdef cseed
        if seed != -1:
            cseed = <int> seed
        else:
            now = datetime.datetime.now()
            cseed = <int> (1000*now.second + now.microsecond/100)

        result = self.c_graph.pySampleSegments(
            &support_points[0, 0],
            &neigh_indices[0, 0],
            &lrf_transforms[0, 0],
            &valid_indices[0],
            &scales[0],
            max_support_point,
            neigh_size,
            neighbors_nb,
            shadowing_threshold,
            cseed,
            feats2d_ptr,
            region_sample_size,
            disconnect_rate,
            &heights[0])

        if result == -1:
            raise NanNormals
        else:
            return support_points, neigh_indices, \
                lrf_transforms.reshape((-1, 3, 3)), valid_indices, \
                np.array(feats2d), scales, heights, bbox

    def visualize(self, int max_support_point=32, float neigh_size=200.,
                  int neighbors_nb=3):
        self.c_graph.vizGraph(max_support_point, neigh_size, neighbors_nb)

    def __dealloc__(self):
        del self.c_graph

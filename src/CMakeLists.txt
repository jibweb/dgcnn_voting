# If the pyx file is a C++ file, we should specify that here.
set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/py_graph_construction.pyx
  PROPERTIES CYTHON_IS_CXX TRUE )

if(APPLE)
	include_directories(/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include/)
endif(APPLE)

# Multi-file cython modules do not appear to be working at the moment.
cython_add_module(py_graph_construction py_graph_construction.pyx py_graph_construction.cpp)


target_link_libraries(py_graph_construction ${PCL_LIBRARIES})
target_link_libraries(py_graph_construction tinyply)

if(${v4r_cad2real_object_classification_ROS})
	set_target_properties(py_graph_construction PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION})
	install(TARGETS py_graph_construction
	  ARCHIVE DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
	  LIBRARY DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
	)
endif(${v4r_cad2real_object_classification_ROS})

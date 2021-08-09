# Defines the CMake commands/policies
cmake_minimum_required(VERSION 2.8.5)

# Set the project name
project(v4r_cad2real_object_classification)

add_compile_options(-std=c++11)

set(CMAKE_BUILD_TYPE Release)

# Make the scripts available in the 'cmake' directory available for the
# 'include()' command, 'find_package()' command.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)

# Include the CMake script UseCython.cmake.  This defines add_cython_module().
# Instruction for use can be found at the top of cmake/UseCython.cmake.
include(UseCython)



execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
    "import numpy as np; print(np.get_include());"
    RESULT_VARIABLE np_result
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE)

# With CMake, a clean separation can be made between the source tree and the
# build tree.  When all source is compiled, as with pure C/C++, the source is
# no-longer needed in the build tree.  However, with pure *.py source, the
# source is processed directly.  To handle this, we reproduce the availability
# of the source files in the build tree.
add_custom_target(ReplicatePythonSourceTree ALL ${CMAKE_COMMAND} -P
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ReplicatePythonSourceTree.cmake
  ${CMAKE_CURRENT_BINARY_DIR}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

find_package(PCL 1.3 REQUIRED COMPONENTS common io filters surface kdtree visualization features)
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

add_library(tinyply SHARED ${v4r_cad2real_object_classification_SOURCE_DIR}/src/tinyply.cpp ${v4r_cad2real_object_classification_SOURCE_DIR}/include/tinyply.h)

include_directories(${v4r_cad2real_object_classification_SOURCE_DIR}/include ${PCL_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(v4r_cad2real_object_classification_PY TRUE)

add_subdirectory(src)

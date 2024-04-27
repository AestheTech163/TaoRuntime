cmake_minimum_required(VERSION 3.17)

project(TaoRuntime LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-fopenmp -mfma")


option(ENABLE_GLOG "Whether to build glog" ON)
option(ENABLE_GFLAGS "Whether to build gflags" ON)
if(ENABLE_GLOG)
    include_directories(${PROJECT_SOURCE_DIR}/third_party/glog/src)
    set(BUILD_TESTING OFF)
    add_subdirectory(third_party/glog)
    include_directories(${glog_BINARY_DIR})
endif()
if(ENABLE_GFLAGS)
    include_directories(${PROJECT_SOURCE_DIR}/third_party/gflags)
    add_subdirectory(third_party/gflags)
    include_directories(${gflags_BINARY_DIR}/include)
endif()


include_directories(${CMAKE_SOURCE_DIR}/include)
file(GLOB src_files src/*)
file(GLOB hdr_files include/*)
set(all_files ${src_files} ${hdr_files})

set(CMAKE_CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
find_package(CUDA REQUIRED)
message(STATUS "cuda version:" ${CUDA_VERSION_STRING})
message(STATUS "cuda include:" ${CUDA_INCLUDE_DIRS})
message(STATUS "cuda libs:" ${CUDA_LIBRARIES})
include_directories(${CUDA_INCLUDE_DIRS})

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS})
set(CMAKE_CUDA_RUNTIME_LIBRARY /usr/local/cuda/lib64)

# set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)


add_executable(test_misc test/test_misc.cpp ${src_files} ${hdr_files})
set_target_properties(test_misc PROPERTIES CUDA_ARCHITECTURES "50;53;60")
target_link_libraries(test_misc cudart)

add_executable(test_tp test/test_tp.cpp)

# add_executable(test_unary test/test_unary.cpp ${all_files} )
# set_target_properties(test_unary PROPERTIES CUDA_ARCHITECTURES "35;50")
# # target_include_directories(test_unary PRIVATE include)
# # target_link_directories(test_unary PRIVATE /usr/local/cuda/lib64)
# target_link_libraries(test_unary cudart)

# link_directories(${PROJECT_BINARY_DIR}/gflags)
# add_executable(test test/test_third.cpp)
# target_link_libraries(test PUBLIC glog gflags)
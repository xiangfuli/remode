cmake_minimum_required(VERSION 3.17)
project(sight_depth_estimation LANGUAGES CXX CUDA)
enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)
message(STATUS "cuda version: " ${CMAKE_CUDA_COMPILER})

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(
    include
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

message(STATUS "Eigen include dirs" ${EIGEN3_INCLUDE_DIRS})

SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

add_library(sight_depth_estimation_cuda STATIC
    src/depthmap_denoiser.cu
    src/reduction.cu
    src/seed_matrix.cu
)

# NCC patch size (must be odd integer, typical values: 3, 5, 7)
ADD_DEFINITIONS(-DRMD_CORR_PATCH_SIDE=5)
# Maximum extent of epipolar line search in pixels
ADD_DEFINITIONS(-DRMD_MAX_EXTENT_EPIPOLAR_SEARCH=100)

add_compile_options(-Wdeprecated-declarations)

add_executable(main src/main.cu src/depthmap.cpp)
target_link_libraries(main sight_depth_estimation_cuda CUDA::cudart ${OpenCV_LIBS})
set_target_properties(main PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)



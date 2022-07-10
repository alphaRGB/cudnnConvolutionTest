project(TEST_CONV2D)
cmake_minimum_required(VERSION 3.10)

enable_language(CXX CUDA)

# set(OpenCV_DIR "/home/wei/ubuntu/Libs/opencv-4.5.1/INSTALL")
set(CMAKE_PREFIX_PATH "/home/penghuiwei/MyWorkspace/Workspace/DockerEnv/Softwares/opencv/INSTALL")
find_package(OpenCV REQUIRED)

# cudnn
set(CUDNN_INCLUDE_DIRS "/usr/include")
set(CUDNN_LIBS "/usr/lib/x86_64-linux-gnu/libcudnn.so")

# cuda
set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")
set(CUDA_LIBS "/usr/local/cuda/lib64")

add_library(conv2d SHARED "conv2d_impl.cu")
target_include_directories(conv2d PRIVATE ${CUDNN_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
target_link_libraries(conv2d ${CUDNN_LIBS} ${CUDA_LIBS}) 

add_executable(main "main.cu")
target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
target_link_libraries(main ${OpenCV_LIBS} ${CUDNN_LIBS} ${CUDA_LIBS} conv2d) 

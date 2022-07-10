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

# set(PYTHON_INCLUDE_DIRS "/home/penghuiwei/MyWorkspace/Workspace/DockerEnv/Softwares/miniconda3/include/python3.8")
# set(PYTHON_LIBS "/home/penghuiwei/MyWorkspace/Workspace/DockerEnv/Softwares/miniconda3/lib/libpython3.so")
# # pybind11
# set(PYBIND_INCLUDE_DIRS "./third_part/pybind11/include")

find_package(PythonLibs REQUIRED)

add_subdirectory(third_part/pybind11)
set(PYBIND11_FINDPYTHON ON)
if(pybind11_FOUND)
    find_package(Python COMPONENTS Interpreter Development)
    message(STATUS "Found pybind11")
    message(STATUS ${pybind11_INCLUDE_DIRS})
    message(STATUS ${pybind11_INCLUDE_DIR})
    message(STATUS ${pybind11_DEFINITIONS})
    message(STATUS ${pybind11_LIBRARIES})
    message(STATUS ${PYTHON_LIBRARIES})
endif(pybind11_FOUND)


# conv lib
add_library(conv2d SHARED "conv2d_impl.cu")
target_include_directories(conv2d PRIVATE ${CUDNN_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
target_link_libraries(conv2d ${CUDNN_LIBS} ${CUDA_LIBS} ${pybind11_LIBRARIES} ${PYTHON_LIBRARIES}) 

# execuable
add_executable(main "main.cu")
target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
target_link_libraries(main ${OpenCV_LIBS} ${CUDNN_LIBS} ${CUDA_LIBS} ${pybind11_LIBRARIES} ${PYTHON_LIBRARIES} conv2d) 

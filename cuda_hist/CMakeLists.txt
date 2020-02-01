cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_CXX_FLAGS "-std=c++11 -pthread -Wall -Werror -O3 ${CMAKE_CXX_FLAGS}")

project(cuda_hist)

find_package(CUDA 10.0 REQUIRED)
find_package(OpenCV REQUIRED)

cuda_add_executable(main src/main_v2.cu)

target_link_libraries(main
  ${OpenCV_LIBS}
)

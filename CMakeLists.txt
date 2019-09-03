cmake_minimum_required(VERSION 3.8)
project(cv411)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV 4.1.1 EXACT REQUIRED PATHS /home/vmachado/OpenCV4.1.1)
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(cv411 main.cpp obj_detect.cpp common.hpp)
target_link_libraries( cv411 ${OpenCV_LIBS} -lpthread)

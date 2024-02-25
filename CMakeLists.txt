cmake_minimum_required(VERSION 2.8)

project(LaneLineProject)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(LaneLineProject src/main.cpp)

target_link_libraries(LaneLineProject ${OpenCV_LIBS} )

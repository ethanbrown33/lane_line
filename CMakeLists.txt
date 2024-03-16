cmake_minimum_required(VERSION 2.8)

project(LaneLineProject)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB HEADERS src/*.h)
file(GLOB SOURCES src/*.cpp)
add_executable(LaneLineProject ${HEADERS} ${SOURCES})

target_link_libraries(LaneLineProject ${OpenCV_LIBS} )

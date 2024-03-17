cmake_minimum_required(VERSION 2.8)

project(LaneLineProject)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB HEADERS src/*.h)
file(GLOB SOURCES src/*.cpp)
add_executable(LaneLineProject ${HEADERS} ${SOURCES})

target_link_libraries(LaneLineProject ${OpenCV_LIBS} )

# Boost needs to be installed
find_package(Boost COMPONENTS system filesystem REQUIRED)
target_link_libraries(LaneLineProject ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY})

# Enable compiler warnings as errors
target_compile_options(LaneLineProject PRIVATE -Werror -Wall -Wextra)

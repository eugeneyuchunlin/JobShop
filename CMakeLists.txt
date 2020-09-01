cmake_minimum_required(VERSION 2.8)
project( Main )
find_package( OpenCV REQUIRED )
set(CMAKE_BUILD_TYPE Debug)

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( Main main.cpp gantt.cpp job.cpp chromosome.cpp machine.cpp configure.cpp)
target_link_libraries( Main ${OpenCV_LIBS} )

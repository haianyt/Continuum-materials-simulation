cmake_minimum_required(VERSION 3.5)
project(mpm)

set(CMAKE_CXX_STANDARD 14)
add_compile_options(-lpthread)
find_package( Threads )
include_directories(eigen)
link_libraries(${X11_LIBRARIES})
include_directories(${X11_INCLUDE_DIR})
include_directories(include)


add_executable(mpm mpm.cpp)
# add_executable(test test.cpp)
target_link_libraries( mpm ${CMAKE_THREAD_LIBS_INIT} )

# list( APPEND CMAKE_PREFIX_PATH "/opt/hfs18.0/toolkit/cmake" )
# find_package(Houdini REQUIRED)
# target_link_libraries(FOSSSim Houdini)
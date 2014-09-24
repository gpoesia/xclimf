cmake_minimum_required(VERSION 2.6)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wextra -O2")
add_library(xclimf xclimf.cpp)
INSTALL(TARGETS xclimf
        ARCHIVE DESTINATION lib)
INSTALL(FILES xclimf.h
        DESTINATION include)

add_executable(example example.cpp)
target_link_libraries(example xclimf)

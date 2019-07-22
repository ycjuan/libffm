cmake_minimum_required(VERSION 3.12)

project(libffm CXX)

OPTION(USE_OPENMP "Enable OpenMP" ON)
OPTION(USE_SSE "Enable SSE acceleration" ON)

set(CMAKE_C_STANDARD 11)
SET(CMAKE_CXX_COMPILER g++)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_FLAGS "-Wall -O3 -std=c++0x -march=native")

if (USE_SSE)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSESSE")
endif(USE_SSE)

if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    # Ignore unknown #pragma warning
    if((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") OR (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    endif()
endif(USE_OPENMP)

add_library(ffm SHARED ffm.cpp timer.cpp)
add_executable(ffm-train ffm-train.cpp ffm.cpp timer.cpp)
add_executable(ffm-predict ffm-predict.cpp ffm.cpp timer.cpp)

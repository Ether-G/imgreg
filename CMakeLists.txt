cmake_minimum_required(VERSION 3.16)
project(imgreg VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -fsanitize=address,undefined")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -DNDEBUG")

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)

# Find FFTW3 using pkg-config
pkg_check_modules(FFTW3 REQUIRED fftw3)

# Check if FFTW threading is available
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_LIBRARIES ${FFTW3_LIBRARIES})
check_cxx_source_compiles(
    "#include <fftw3.h>
     int main() { fftw_init_threads(); return 0; }"
    FFTW_THREADS_AVAILABLE
)

if(FFTW_THREADS_AVAILABLE)
    add_definitions(-DFFTW_THREADS_AVAILABLE)
endif()

find_package(OpenMP)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${FFTW3_INCLUDE_DIRS})

# Source files
set(SOURCES
    src/main.cpp
    src/image_correlator.cpp
    src/spatial_correlator.cpp
    src/fft_correlator.cpp
    src/pearson_correlator.cpp
    src/performance_analyzer.cpp
    src/thread_manager.cpp
    src/memory_pool.cpp
    src/generate_shifted.cpp
)

# Create library (without main.cpp)
set(LIB_SOURCES
    src/image_correlator.cpp
    src/spatial_correlator.cpp
    src/fft_correlator.cpp
    src/pearson_correlator.cpp
    src/performance_analyzer.cpp
    src/thread_manager.cpp
    src/memory_pool.cpp
)

# Create library target
add_library(imgreg_lib ${LIB_SOURCES})

# Link libraries to the library target
target_link_libraries(imgreg_lib 
    ${OpenCV_LIBS}
    ${FFTW3_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
)

# OpenMP support for library
if(OpenMP_CXX_FOUND)
    target_link_libraries(imgreg_lib OpenMP::OpenMP_CXX)
endif()

# Create executable
add_executable(imgreg src/main.cpp)

# Link executable with library
target_link_libraries(imgreg imgreg_lib)

# Add generate_shifted utility
add_executable(generate_shifted src/generate_shifted.cpp)
target_link_libraries(generate_shifted ${OpenCV_LIBS})

# Install target
install(TARGETS imgreg imgreg_lib generate_shifted DESTINATION bin)

# Add test directory if it exists
if(EXISTS "${CMAKE_SOURCE_DIR}/tests")
    add_subdirectory(tests)
endif() 
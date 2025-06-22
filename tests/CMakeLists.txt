# Test configuration
enable_testing()

# Add test executable
add_executable(imgreg_test test_main.cpp)

# Include directories
target_include_directories(imgreg_test PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Link with main library and required libraries
target_link_libraries(imgreg_test 
    imgreg_lib
    ${OpenCV_LIBS}
    ${FFTW3_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
)

# OpenMP support
if(OpenMP_CXX_FOUND)
    target_link_libraries(imgreg_test OpenMP::OpenMP_CXX)
endif()

# Add test
add_test(NAME BasicTest COMMAND imgreg_test) 
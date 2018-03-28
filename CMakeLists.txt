cmake_minimum_required(VERSION 3.9)
project(SinewaveFitterTuning)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCL REQUIRED)
find_package(Boost COMPONENTS chrono random date_time REQUIRED)

add_subdirectory(minusDarwin)
add_subdirectory(synthSignal)

add_executable(SinewaveFitterTuning main.cpp)
target_include_directories(SinewaveFitterTuning PUBLIC minusDarwin synthSignal ${OpenCL_INCLUDE_DIR} ${Boost_INCLUDE_DIR})
target_link_libraries(SinewaveFitterTuning PUBLIC MinusDarwin SynthSignal OpenCL::OpenCL ${Boost_LIBRARIES} -lpthread -lm)
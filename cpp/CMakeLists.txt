cmake_minimum_required(VERSION 3.20)

project(ButterflyCServer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)
find_path(TENSORFLOW_INCLUDE_DIR tensorflow/c/c_api.h HINTS /usr/local/include)
find_library(TENSORFLOW_LIBRARY tensorflow HINTS /usr/local/lib)
find_library(
    TENSORFLOW_FRAMEWORK_LIBRARY
    tensorflow_framework
    HINTS /usr/local/lib
)

if(NOT TENSORFLOW_INCLUDE_DIR OR NOT TENSORFLOW_LIBRARY)
    message(FATAL_ERROR "TensorFlow C API not found. Install libtensorflow under /usr/local.")
endif()

add_executable(butterflyc_server src/server.cpp)
target_compile_definitions(butterflyc_server PRIVATE CROW_USE_BOOST=1)
target_include_directories(
    butterflyc_server
    PRIVATE
        ${PROJECT_SOURCE_DIR}/../app
        ${TENSORFLOW_INCLUDE_DIR}
)
target_link_directories(
    butterflyc_server
    PRIVATE
        /usr/local/lib
)
target_link_libraries(
    butterflyc_server
    PRIVATE
        Threads::Threads
        ${TENSORFLOW_LIBRARY}
        ${TENSORFLOW_FRAMEWORK_LIBRARY}
)

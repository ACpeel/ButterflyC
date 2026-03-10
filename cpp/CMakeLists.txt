cmake_minimum_required(VERSION 3.20)

project(ButterflyCServer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenCV REQUIRED)
find_package(Crow CONFIG QUIET)

set(CROW_TARGET "")
set(CROW_INCLUDE_DIR "")

if(TARGET Crow::Crow)
    set(CROW_TARGET Crow::Crow)
elseif(TARGET Crow)
    set(CROW_TARGET Crow)
elseif(TARGET crow)
    set(CROW_TARGET crow)
else()
    find_path(CROW_INCLUDE_DIR crow.h HINTS /usr/local/include /usr/include)
    if(NOT CROW_INCLUDE_DIR)
        message(
            FATAL_ERROR
            "Crow headers not found. Install Crow or provide CrowConfig.cmake before building cpp/."
        )
    endif()
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h HINTS /opt/onnxruntime/include /usr/local/include)
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS /opt/onnxruntime/lib /usr/local/lib)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(
        FATAL_ERROR
        "ONNX Runtime not found. Install onnxruntime and set include/lib paths."
    )
endif()

add_executable(butterflyc_server src/server.cpp)
target_compile_definitions(butterflyc_server PRIVATE CROW_USE_BOOST=1)
target_include_directories(
    butterflyc_server
    PRIVATE
        ${CROW_INCLUDE_DIR}
        ${ONNXRUNTIME_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(
    butterflyc_server
    PRIVATE
        ${CROW_TARGET}
        Boost::system
        Threads::Threads
        ${ONNXRUNTIME_LIBRARY}
        ${OpenCV_LIBS}
)

cmake_minimum_required(VERSION 3.5)

project(DeepSORT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package (Eigen3 3.3 REQUIRED NO_MODULE)


set(ONNXRUNTIME_DIR "D:/doorw_source/onnxruntime-win-x64-gpu-1.20.1")

include_directories("${ONNXRUNTIME_DIR}/include"
    )
find_package(OpenCV 4 REQUIRED )


include_directories(
    ${OpenCV_INCLUDE_DIRS}/include
    ${CMAKE_SOURCE_DIR}/tracker/deepsort/include
#    ${CMAKE_SOURCE_DIR}/tracker/bytetrack/include
#    ${CMAKE_SOURCE_DIR}/detector/YOLOv5/include
    )
message(STATUS "-OpenCV_INCLUDE_DIRS--- = ${OpenCV_INCLUDE_DIRS}/include")
# -------------- Compile CUDA for FP16 inference if needed  ------------------#
option(USE_CUDA "Enable CUDA support" ON)
if (NOT APPLE AND USE_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    add_definitions(-DUSE_CUDA)
else ()
    set(USE_CUDA OFF)
endif ()

# -------------- ONNXRUNTIME  ------------------#
# -------------- ONNXRUNTIME  ------------------#

# Set ONNXRUNTIME_VERSION
set(ONNXRUNTIME_VERSION 1.20.1)

if (WIN32)
    if (USE_CUDA)
	message(STATUS "---- = ${CMAKE_CURRENT_SOURCE_DIR}")
	message(STATUS "---- = ${ONNXRUNTIME_VERSION}")
        set(ONNXRUNTIME_ROOT "D:/doorw_source/onnxruntime-win-x64-gpu-1.20.1")
    else ()
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}")
    endif ()
elseif (LINUX)
    if (USE_CUDA)
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}")
    else ()
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}")
    endif ()
elseif (APPLE)
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-osx-arm64-${ONNXRUNTIME_VERSION}")
    # Apple X64 binary
    # set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-osx-x64-${ONNXRUNTIME_VERSION}")
    # Apple Universal binary
    # set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-osx-universal2-${ONNXRUNTIME_VERSION}")
else ()
    message(SEND_ERROR "Variable ONNXRUNTIME_ROOT is not set properly. Please check if your cmake project \
    is not compiled with `-D WIN32=TRUE`, `-D LINUX=TRUE`, or `-D APPLE=TRUE`!")
endif ()

include_directories(${PROJECT_NAME} ${ONNXRUNTIME_ROOT}/include)


add_executable(DeepSORT
    #detector/YOLOv5/src/YOLOv5Detector.cpp

    tracker/deepsort/src/inference.cpp
    tracker/deepsort/src/model.cpp
    tracker/deepsort/src/kalmanfilter.cpp
    tracker/deepsort/src/linear_assignment.cpp
    tracker/deepsort/src/nn_matching.cpp
    tracker/deepsort/src/track.cpp
    tracker/deepsort/src/tracker.cpp
    tracker/deepsort/src/munkres.cpp
    tracker/deepsort/src/hungarianoper.cpp

    #tracker/bytetrack/src/BytekalmanFilter.cpp
    #tracker/bytetrack/src/BYTETracker.cpp
    #tracker/bytetrack/src/lapjv.cpp
    #tracker/bytetrack/src/STrack.cpp
    #tracker/bytetrack/src/utils.cpp

    main.cpp)
target_link_libraries(DeepSORT PRIVATE "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib" ${OpenCV_LIBS} Eigen3::Eigen)




#add_executable(test_detector
#    detector/YOLOv5/src/YOLOv5Detector.cpp

#    test_detector.cpp)
#target_link_libraries(test_detector PRIVATE  ${OpenCV_LIBS} Eigen3::Eigen)

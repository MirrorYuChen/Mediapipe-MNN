cmake_minimum_required(VERSION 3.1)

project(Mediapipe-MNN)

set(3RD_DIR ${CMAKE_CURRENT_LIST_DIR}/3rdLibs)
set(SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/source)

add_subdirectory(${3RD_DIR}/MNN-2.3.0)

include_directories(
    ${SRC_DIR}
    ${3RD_DIR}/MNN-2.3.0/include
)

if(MSVC)
    set(OpenCV_LIBS opencv_world342)
    include_directories(${3RD_DIR}/opencv-3.4.2/include)
    link_directories(${CMAKE_CURRENT_LIST_DIR}/3rdLibs/opencv-3.4.2/lib)
    list(APPEND OBJECTS_TO_LINK opencv_world342)
else(MSVC)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    list(APPEND OBJECTS_TO_LINK ${OpenCV_LIBS})
endif(MSVC)

file(GLOB_RECURSE SRC_FILES
    ${SRC_DIR}/*.h
    ${SRC_DIR}/*.cc
)

add_library(${PROJECT_NAME} STATIC ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} MNN ${OpenCV_LIBS})

set(EXAMPLE_LIST
    TestPalmDetection
    TestFaceDetection
    TestFaceDetectorLite
    TestFaceDetectorYN
    TestImageEmbedding
    TestFeatureDetector
    TestPoseDetector
    TestClassifier
    TestSegmentDetector
)

foreach(EXAMPLE ${EXAMPLE_LIST})
    add_executable(${EXAMPLE} examples/${EXAMPLE}.cc)
    target_link_libraries(${EXAMPLE} ${PROJECT_NAME} ${OBJECTS_TO_LINK})
endforeach(EXAMPLE ${EXAMPLE_LIST})

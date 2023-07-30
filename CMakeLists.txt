cmake_minimum_required(VERSION 3.1)

project(Mediapipe-MNN)

set(3RD_DIR ${CMAKE_CURRENT_LIST_DIR}/3rdLibs)
set(SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/source)

add_subdirectory(${3RD_DIR}/MNN-2.3.0)

include_directories(
    ${SRC_DIR}
    ${3RD_DIR}/MNN-2.3.0/include
    ${3RD_DIR}/opencv-3.4.2/include
)
link_directories(${CMAKE_CURRENT_LIST_DIR}/3rdLibs/opencv-3.4.2/lib)
list(APPEND OBJECTS_TO_LINK opencv_world342)

file(GLOB_RECURSE SRC_FILES
    ${SRC_DIR}/*.h
    ${SRC_DIR}/*.cc
)

add_library(${PROJECT_NAME} STATIC ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} MNN opencv_world342)

set(EXAMPLE_LIST
    TestPalmDetection
    TestFaceDetection
)

foreach(EXAMPLE ${EXAMPLE_LIST})
    add_executable(${EXAMPLE} examples/${EXAMPLE}.cc)
    target_link_libraries(${EXAMPLE} ${PROJECT_NAME} ${OBJECTS_TO_LINK})
endforeach(EXAMPLE ${EXAMPLE_LIST})

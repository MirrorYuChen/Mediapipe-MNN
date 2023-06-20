cmake_minimum_required(VERSION 3.1)

project(Mediapipe-Hand)

set(3RD_DIR ${CMAKE_CURRENT_LIST_DIR}/3rdLibs)
set(SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/source)

add_subdirectory(${3RD_DIR}/MNN-2.3.0)

include_directories(
    ${SRC_DIR}
    ${3RD_DIR}/MNN-2.3.0/include
)

file(GLOB_RECURSE SRC_FILES
    ${SRC_DIR}/*.h
    ${SRC_DIR}/*.cc
)

add_executable(TestHand ${SRC_FILES})
target_link_libraries(TestHand MNN) 


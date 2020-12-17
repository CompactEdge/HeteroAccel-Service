# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "human_pose_estimation_demo1")

# Find OpenCV components if exist
find_package(OpenCV COMPONENTS highgui QUIET)
if(NOT(OpenCV_FOUND))
    message(WARNING "OPENCV is disabled or not found, " ${TARGET_NAME} " skipped")
    return()
endif()
message("dir:" ${CMAKE_CURRENT_SOURCE_DIR})
#target_include_directories(${CMAKE_CURRENT_SOURCE_DIR})
set(SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/include/human_pose.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/human_pose_estimation_demo.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/human_pose_estimator.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/peak.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/render_human_pose.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/human_pose.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/human_pose_estimator.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/peak.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/render_human_pose.cpp
            #${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
            )

set(libevent_dir "/home/kpst/Documents/libevent-2.0.21-stable/.libs")
add_executable(${TARGET_NAME} main.cpp ${SOURCES})

link_directories(${libevent_dir})
#link_directories(${LIB_FOLDER})

set_target_properties(${TARGET_NAME} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE"
  COMPILE_PDB_NAME ${TARGET_NAME})

target_include_directories(${TARGET_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_link_libraries(${TARGET_NAME} ${InferenceEngine_LIBRARIES} ${OpenCV_LIBS} gflags)

if(UNIX)
  target_link_libraries(${TARGET_NAME} ${LIB_DL} pthread event)
endif()

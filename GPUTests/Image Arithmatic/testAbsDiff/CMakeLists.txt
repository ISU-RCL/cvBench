#
#     	(c) Copyright 2019 Iowa State University.
#     	All rights reserved.
#	
#     	Author: Murad Qasaimeh <qasaimeh@iastate.edu>
#     	Date:   2018/03/27 

# cmake needs this line
cmake_minimum_required(VERSION 2.8)

set(subdirLevels "../../..")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/${subdirLevels}/PYNQ-ComputerVision/frameworks/cmakeModules)

# Define project name
SET (ProjectName testGPUAbsDiff)
project(${ProjectName})


# Find packages. 
# OpenCV uses environment variable OpenCV_DIR
find_package(OpenCV REQUIRED)
find_package(VisionWorks REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# add macro for WITH_CUDA option
OPTION(WITH_CUDA "Compile with CUDA" OFF) # Disabled by default
if (WITH_CUDA)
	add_definitions(-DWITH_CUDA)
	message(STATUS "INFO: setting WITH_CUDA directive")
endif (WITH_CUDA)

# add macro for WITH_OPENCL option
OPTION(WITH_OPENVX "Compile with OpenVX" OFF) # Disabled by default
if (WITH_OPENVX)
	add_definitions(-DWITH_OPENVX)
	message(STATUS "INFO: setting WITH_OPENVX directive")
endif (WITH_OPENVX)

add_definitions(-DIVX_USE_OPENCV)
# add subdirectories 
add_subdirectory(${PROJECT_SOURCE_DIR}/../../../PYNQ-ComputerVision/frameworks/utilities/OpenCVUtils ${CMAKE_CURRENT_BINARY_DIR}/OpenCVUtils)
add_subdirectory(${PROJECT_SOURCE_DIR}/../../../PYNQ-ComputerVision/frameworks/utilities/HRTimer ${CMAKE_CURRENT_BINARY_DIR}/HRTimer)
include_directories(
	${PROJECT_SOURCE_DIR}/inc 
	${PROJECT_SOURCE_DIR}/../../../PYNQ-ComputerVision/frameworks/utilities/OpenCVUtils/inc
	${OpenCV_INCLUDE_DIRS}	
)

# ---- goldenOpenCVAbsDiff ----
SET (currentTarget testGPUAbsDiff)

add_executable(${currentTarget} src/testGPUAbsDiff.cpp)

target_link_libraries(${currentTarget} 
	OpenCVUtils 
	HRTimer
	${OpenCV_LIBS}
	${VisionWorks_LIBRARIES} 
	${VisionWorks-NVXIO_LIBRARIES}
	)
	

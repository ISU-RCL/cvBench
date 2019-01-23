/*****************************************************************************
*
* 	Copyright (c) 2019 Iowa State University
*
* 	Redistribution and use in source and binary forms, with or without
* 	modification, are permitted provided that the following conditions 
* 	are met:
*
* 	1.Redistributions of source code must retain the above copyright 
* 	notice, this list of conditions and the following disclaimer.
* 	2.Redistributions in binary form must reproduce the above copyright 
* 	notice, this list of conditions and the following disclaimer in the
* 	documentation and/or other materials provided with the distribution.
* 	3.Neither the name of the copyright holder nor the names of its 
* 	contributors may be used to endorse or promote products derived 
* 	from this software without specific prior written permission.
*
* 	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* 	"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* 	LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
* 	FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
*	COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
* 	INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
* 	BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
* 	LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* 	CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
* 	LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
* 	ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
* 	POSSIBILITY OF SUCH DAMAGE.
*
*     	(c) Copyright 2019 Iowa State University.
*     	All rights reserved.
*	
*     	Author: Murad Qasaimeh <qasaimeh@iastate.edu>
*     	Date:   2018/03/27
*****************************************************************************/

# cmake needs this line
cmake_minimum_required(VERSION 2.8)

set(subdirLevels "../../..")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/${subdirLevels}/PYNQ-ComputerVision/frameworks/cmakeModules)

# Define project name
SET (ProjectName testGPUFilter2D)
project(${ProjectName})



message(STATUS "    version: ${OpenCV_VERSION}")

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

# add macro for WITH_OpenVX option
OPTION(WITH_OPENVX "Compile with OpenVX" OFF) # Disabled by default
if (WITH_OPENVX)
	add_definitions(-DWITH_OPENVX)
	message(STATUS "INFO: setting WITH_OPENVX directive")
endif (WITH_OPENVX)

add_definitions(-DIVX_USE_OPENCV)
# add subdirectories
add_subdirectory(${PROJECT_SOURCE_DIR}/../../PYNQ-ComputerVision/frameworks/utilities/OpenCVUtils ${CMAKE_CURRENT_BINARY_DIR}/OpenCVUtils)
add_subdirectory(${PROJECT_SOURCE_DIR}/../../PYNQ-ComputerVision/frameworks/utilities/HRTimer ${CMAKE_CURRENT_BINARY_DIR}/HRTimer)
include_directories(
	${PROJECT_SOURCE_DIR}/inc 
	${PROJECT_SOURCE_DIR}/../../PYNQ-ComputerVision/frameworks/utilities/OpenCVUtils/inc
	${OpenCV_INCLUDE_DIRS}	
)

# ---- goldenOpenCVFilter2D ----
SET (currentTarget testGPUFilter2D)

add_executable(${currentTarget} src/testGPUFilter2D.cpp)

target_link_libraries(${currentTarget} 
	OpenCVUtils 
	HRTimer
	${OpenCV_LIBS}
	${VisionWorks_LIBRARIES} 
	${VisionWorks-NVXIO_LIBRARIES}
	)
	

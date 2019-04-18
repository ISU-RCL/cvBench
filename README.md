## Benchmarking Analysis of Vision Kernels on Embedded CPU, GPU and FPGA :

<p align="justify">

This repository contains benchmark framework for measuring and comparing energy efficiency of different vision kernels on embedded platforms. It aims to provide computer vision community an easy tool to analyze the performance of vision kernels on different hardware architectures and aids with determining which hardware architecture is most suitable for different kind of vision applications.

</p>

Table of Contents:
* [Repository structure](#Repository_structure) 
* [Hardware and Software Environments](#Hardware_and_Software_Environments)
* [List of Vision Kernels](#List_of_Vision_Kernels)
* [Installation](#Installation) 
* [Build Test Codes](#Build_Test_Codes) 
* [Results Summary](#Results_Summary) 
* [References](#references)
* [License](#license) 
 
 

## Repository structure

This repository consists of two parts:
* FPGA testbenches:
* GPU testbenches:
.
├── FPGATests
│   └── README.md
├── GPUTests
│   ├── Geometric Transforms
│   │   ├── testRemap
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPURemap.cpp
│   │   ├── testResize
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPUResize.cpp
│   │   ├── testWarpAffine
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPUWarpAffine.cpp
│   │   └── testWarpPerspective
│   │       ├── CMakeLists.txt
│   │       └── src
│   │           └── testGPUWarpPerspective.cpp
│   ├── Image Analysis
│   │   ├── testCalcHist
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPUCalcHist.cpp
│   │   ├── testEqualizeHist
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPUEqualizeHist.cpp
│   │   ├── testIntegral
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPUIntegral.cpp
│   │   ├── testMeanStdDev
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPUMeanStdDev.cpp
│   │   └── testMinMaxLoc
│   │       ├── CMakeLists.txt
│   │       └── src
│   │           └── testGPUMinMaxLoc.cpp
│   ├── Image Arithmatic
│   │   ├── testAbsDiff
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPUAbsDiff.cpp
│   │   ├── testAccumulate
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPUAccumulate.cpp
│   │   ├── testAccumulateSquare
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPUAccumulateSquare.cpp
│   │   ├── testAccumulateWeighted
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPUAccumulateWeighted.cpp
│   │   ├── testAdd
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPUAdd.cpp
│   │   ├── testBitwise_and
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPUBitwise_and.cpp
│   │   ├── testBitwise_not
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPUBitwise_not.cpp
│   │   ├── testBitwise_or
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPUBitwise_or.cpp
│   │   ├── testBitwise_xor
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPUBitwise_xor.cpp
│   │   ├── testMagnitude
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPUMagnitude.cpp
│   │   ├── testMultiply
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPUMultiply.cpp
│   │   ├── testPhase
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPUPhase.cpp
│   │   ├── testSubtract
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPUSubtract.cpp
│   │   └── testThreshold
│   │       ├── CMakeLists.txt
│   │       └── src
│   │           └── testGPUThreshold.cpp
│   ├── Image Features
│   │   ├── testCanny
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPUCanny.cpp
│   │   ├── testCornerHarris
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPUCornerHarris.cpp
│   │   └── testFast
│   │       ├── CMakeLists.txt
│   │       └── src
│   │           └── testGPUFast.cpp
│   ├── Image Filters
│   │   ├── testBoxFilter
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPUBoxFilter.cpp
│   │   ├── testDilate
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPUDilate.cpp
│   │   ├── testErode
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPUErode.cpp
│   │   ├── testFilter2D
│   │   │   ├── build
│   │   │   │   ├── CMakeCache.txt
│   │   │   │   └── CMakeFiles
│   │   │   │       ├── 3.3.2
│   │   │   │       │   ├── CMakeCCompiler.cmake
│   │   │   │       │   ├── CMakeCXXCompiler.cmake
│   │   │   │       │   ├── CMakeDetermineCompilerABI_C.bin
│   │   │   │       │   ├── CMakeDetermineCompilerABI_CXX.bin
│   │   │   │       │   ├── CMakeSystem.cmake
│   │   │   │       │   ├── CompilerIdC
│   │   │   │       │   │   ├── a.out
│   │   │   │       │   │   └── CMakeCCompilerId.c
│   │   │   │       │   └── CompilerIdCXX
│   │   │   │       │       ├── a.out
│   │   │   │       │       └── CMakeCXXCompilerId.cpp
│   │   │   │       ├── cmake.check_cache
│   │   │   │       ├── CMakeOutput.log
│   │   │   │       ├── CMakeTmp
│   │   │   │       ├── feature_tests.bin
│   │   │   │       ├── feature_tests.c
│   │   │   │       └── feature_tests.cxx
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPUFilter2D.cpp
│   │   ├── testMedianBlur
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPUMedian.cpp
│   │   ├── testPyrDown
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPUPyrDown.cpp
│   │   └── testPyrUp
│   │       ├── CMakeLists.txt
│   │       └── src
│   │           └── testGPUPyrUp.cpp
│   ├── Input Processing
│   │   ├── testCombine
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPUCombine.cpp
│   │   ├── testConvertTo
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPUConvertTo.cpp
│   │   ├── testCvtColor
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPUCvtColor.cpp
│   │   ├── testExtract
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPUSplit.cpp
│   │   └── testLUT
│   │       ├── CMakeLists.txt
│   │       └── src
│   │           └── testGPULUT.cpp
│   ├── Optical Flow & Depth
│   │   ├── testOpticalFlowPyr
│   │   │   ├── CMakeLists.txt
│   │   │   └── src
│   │   │       └── testGPUOpticalFlowPyr.cpp
│   │   └── testStereoBM
│   │       ├── CMakeLists.txt
│   │       └── src
│   │           └── testGPUStereoBM.cpp
│   └── README.md
├── LICENSE
├── PYNQ-ComputerVision
│   ├── applicationCode
│   │   ├── examples
│   │   │   └── cvToFromXf
│   │   │       ├── CMakeLists.txt
│   │   │       └── src
│   │   │           └── cvToFromXf.cpp
│   │   ├── overlayTests
│   │   │   └── testPython
│   │   │       ├── testXfFilter2D_HDMI.py
│   │   │       └── testXfFilter2D.py
│   │   └── unitTests
│   │       └── testPython
│   │           ├── OpenCVUtils.py
│   │           ├── runUnitTests.py
│   │           ├── testXfBitwise_and.py
│   │           ├── testXfBitwise_not.py
│   │           ├── testXfBitwise_or.py
│   │           ├── testXfBitwise_xor.py
│   │           ├── testXfBoxFilter.py
│   │           ├── testXfCanny.py
│   │           ├── testXfDilate.py
│   │           ├── testXfErode.py
│   │           ├── testXfFilter2D.py
│   │           ├── testXfMedianBlur.py
│   │           ├── testXfRemap.py
│   │           ├── testXfResize.py
│   │           ├── testXfStereoBM.py
│   │           ├── testXfSubtract.py
│   │           └── testXfThreshold.py
│   ├── block_diagram.png
│   ├── boards
│   │   ├── Pynq-Z1
│   │   │   ├── notebooks
│   │   │   │   ├── filter2d_and_dilate.ipynb
│   │   │   │   ├── filter2d_and_remap.ipynb
│   │   │   │   └── filter2d.ipynb
│   │   │   └── overlays
│   │   │       ├── __init__.py
│   │   │       ├── xv2Filter2DDilate.bit
│   │   │       ├── xv2Filter2DDilate.hwh
│   │   │       ├── xv2Filter2DDilate.so
│   │   │       ├── xv2Filter2DRemap.bit
│   │   │       ├── xv2Filter2DRemap.hwh
│   │   │       └── xv2Filter2DRemap.so
│   │   ├── Pynq-Z2
│   │   │   ├── notebooks
│   │   │   │   ├── filter2d_and_dilate.ipynb -> ../../Pynq-Z1/notebooks/filter2d_and_dilate.ipynb
│   │   │   │   ├── filter2d_and_remap.ipynb -> ../../Pynq-Z1/notebooks/filter2d_and_remap.ipynb
│   │   │   │   └── filter2d.ipynb -> ../../Pynq-Z1/notebooks/filter2d.ipynb
│   │   │   └── overlays
│   │   │       ├── __init__.py
│   │   │       ├── xv2Filter2DDilate.bit
│   │   │       ├── xv2Filter2DDilate.hwh
│   │   │       ├── xv2Filter2DDilate.so
│   │   │       ├── xv2Filter2DRemap.bit
│   │   │       ├── xv2Filter2DRemap.hwh
│   │   │       └── xv2Filter2DRemap.so
│   │   ├── Ultra96
│   │   │   ├── notebooks
│   │   │   │   ├── 0__Jupyter_Notebooks_walkthrough.ipynb
│   │   │   │   ├── 1__Intro_to_OpenCV_on_Jupyter_notebooks.ipynb
│   │   │   │   ├── 2__Overlays-filter2d_and_dilate.ipynb
│   │   │   │   ├── 3a_Overlays-stereo_block_matching.ipynb
│   │   │   │   ├── 4__Overlays-opticalflow.ipynb
│   │   │   │   ├── 5__Build_your_own-tracking_example.ipynb
│   │   │   │   └── images
│   │   │   │       ├── 0000000015_0106_extract_L.png
│   │   │   │       ├── 0000000015_0106_extract_R.png
│   │   │   │       ├── 0000000015_0106_sync_L.png
│   │   │   │       ├── 0000000015_0106_sync_R.png
│   │   │   │       ├── 0000000020_0106_extract_L.png
│   │   │   │       ├── 0000000020_0106_extract_R.png
│   │   │   │       ├── 000005_10_L.png
│   │   │   │       ├── 000005_10_R.png
│   │   │   │       ├── imL.png
│   │   │   │       ├── imR.png
│   │   │   │       └── vtest
│   │   │   │           ├── scene00002.png
│   │   │   │           ├── scene00003.png
│   │   │   │           ├── scene00004.png
│   │   │   │           └── scene00005.png
│   │   │   └── overlays
│   │   │       ├── __init__.py
│   │   │       ├── xv2CalcOpticalFlowDenseNonPyrLK.bit
│   │   │       ├── xv2CalcOpticalFlowDenseNonPyrLK.hwh
│   │   │       ├── xv2CalcOpticalFlowDenseNonPyrLK.so
│   │   │       ├── xv2Filter2DDilateAbsdiff.bit
│   │   │       ├── xv2Filter2DDilateAbsdiff.hwh
│   │   │       ├── xv2Filter2DDilateAbsdiff.so
│   │   │       ├── xv2Filter2DDilate.bit
│   │   │       ├── xv2Filter2DDilate.hwh
│   │   │       ├── xv2Filter2DDilate.so
│   │   │       ├── xv2Filter2DDilateThreshold.bit
│   │   │       ├── xv2Filter2DDilateThreshold.hwh
│   │   │       ├── xv2Filter2DDilateThreshold.so
│   │   │       ├── xv2RemapStereoBM.bit
│   │   │       ├── xv2RemapStereoBM.hwh
│   │   │       └── xv2RemapStereoBM.so
│   │   └── ZCU104
│   │       ├── notebooks
│   │       │   ├── filter2d_and_dilate.ipynb -> ../../Pynq-Z1/notebooks/filter2d_and_dilate.ipynb
│   │       │   ├── filter2d_and_remap.ipynb -> ../../Pynq-Z1/notebooks/filter2d_and_remap.ipynb
│   │       │   └── filter2d.ipynb -> ../../Pynq-Z1/notebooks/filter2d.ipynb
│   │       └── overlays
│   │           └── __init__.py
│   ├── components
│   │   ├── bitwise_and
│   │   │   └── xfSDxKernel
│   │   │       ├── inc
│   │   │       │   ├── PythonBindingXfSDxBitwise_and.h
│   │   │       │   ├── xfBitwise_andCoreForVivadoHLS.h.in
│   │   │       │   └── xfSDxBitwise_and.h
│   │   │       └── src
│   │   │           ├── xfBitwise_andCoreForVivadoHLS.cpp.in
│   │   │           └── xfBitwise_and.cpp.in
│   │   ├── bitwise_not
│   │   │   └── xfSDxKernel
│   │   │       ├── inc
│   │   │       │   ├── PythonBindingXfSDxBitwise_not.h
│   │   │       │   ├── xfBitwise_notCoreForVivadoHLS.h.in
│   │   │       │   └── xfSDxBitwise_not.h
│   │   │       └── src
│   │   │           ├── xfBitwise_notCoreForVivadoHLS.cpp.in
│   │   │           └── xfBitwise_not.cpp.in
│   │   ├── bitwise_or
│   │   │   └── xfSDxKernel
│   │   │       ├── inc
│   │   │       │   ├── PythonBindingXfSDxBitwise_or.h
│   │   │       │   ├── xfBitwise_orCoreForVivadoHLS.h.in
│   │   │       │   └── xfSDxBitwise_or.h
│   │   │       └── src
│   │   │           ├── xfBitwise_orCoreForVivadoHLS.cpp.in
│   │   │           └── xfBitwise_or.cpp.in
│   │   ├── bitwise_xor
│   │   │   └── xfSDxKernel
│   │   │       ├── inc
│   │   │       │   ├── PythonBindingXfSDxBitwise_xor.h
│   │   │       │   ├── xfBitwise_xorCoreForVivadoHLS.h.in
│   │   │       │   └── xfSDxBitwise_xor.h
│   │   │       └── src
│   │   │           ├── xfBitwise_xorCoreForVivadoHLS.cpp.in
│   │   │           └── xfBitwise_xor.cpp.in
│   │   ├── boxFilter
│   │   │   └── xfSDxKernel
│   │   │       ├── inc
│   │   │       │   ├── PythonBindingXfSDxBoxFilter.h
│   │   │       │   ├── xfBoxFilterCoreForVivadoHLS.h.in
│   │   │       │   └── xfSDxBoxFilter.h
│   │   │       └── src
│   │   │           ├── xfBoxFilterCoreForVivadoHLS.cpp.in
│   │   │           └── xfBoxFilter.cpp.in
│   │   ├── canny
│   │   │   └── xfSDxKernel
│   │   │       ├── inc
│   │   │       │   ├── PythonBindingXfSDxCanny.h
│   │   │       │   ├── xfCannyCoreForVivadoHLS.h.in
│   │   │       │   └── xfSDxCanny.h
│   │   │       └── src
│   │   │           ├── xfCannyCoreForVivadoHLS.cpp.in
│   │   │           └── xfCanny.cpp.in
│   │   ├── dilate
│   │   │   └── xfSDxKernel
│   │   │       ├── inc
│   │   │       │   ├── PythonBindingXfSDxDilate.h
│   │   │       │   ├── xfDilateCoreForVivadoHLS.h.in
│   │   │       │   └── xfSDxDilate.h
│   │   │       └── src
│   │   │           ├── xfDilateCoreForVivadoHLS.cpp.in
│   │   │           └── xfDilate.cpp.in
│   │   ├── erode
│   │   │   └── xfSDxKernel
│   │   │       ├── inc
│   │   │       │   ├── PythonBindingXfSDxErode.h
│   │   │       │   ├── xfErodeCoreForVivadoHLS.h.in
│   │   │       │   └── xfSDxErode.h
│   │   │       └── src
│   │   │           ├── xfErodeCoreForVivadoHLS.cpp.in
│   │   │           └── xfErode.cpp.in
│   │   ├── filter2D
│   │   │   └── xfSDxKernel
│   │   │       ├── inc
│   │   │       │   ├── PythonBindingXfSDxFilter2D.h
│   │   │       │   ├── xfFilter2DCoreForVivadoHLS.h.in
│   │   │       │   └── xfSDxFilter2D.h
│   │   │       └── src
│   │   │           ├── xfFilter2DCoreForVivadoHLS.cpp.in
│   │   │           └── xfFilter2D.cpp.in
│   │   ├── medianBlur
│   │   │   └── xfSDxKernel
│   │   │       ├── inc
│   │   │       │   ├── PythonBindingXfSDxMedianBlur.h
│   │   │       │   ├── xfMedianBlurCoreForVivadoHLS.h.in
│   │   │       │   └── xfSDxMedianBlur.h
│   │   │       └── src
│   │   │           ├── xfMedianBlurCoreForVivadoHLS.cpp.in
│   │   │           └── xfMedianBlur.cpp.in
│   │   ├── remap
│   │   │   └── xfSDxKernel
│   │   │       ├── inc
│   │   │       │   ├── PythonBindingXfSDxRemap.h
│   │   │       │   ├── xfRemapCoreForVivadoHLS.h.in
│   │   │       │   └── xfSDxRemap.h
│   │   │       └── src
│   │   │           ├── xfRemapCoreForVivadoHLS.cpp.in
│   │   │           └── xfRemap.cpp.in
│   │   ├── resize
│   │   │   └── xfSDxKernel
│   │   │       ├── inc
│   │   │       │   ├── PythonBindingXfSDxResize.h
│   │   │       │   ├── xfResizeCoreForVivadoHLS.h.in
│   │   │       │   └── xfSDxResize.h
│   │   │       └── src
│   │   │           ├── xfResizeCoreForVivadoHLS.cpp.in
│   │   │           └── xfResize.cpp.in
│   │   ├── stereoBM
│   │   │   └── xfSDxKernel
│   │   │       ├── inc
│   │   │       │   ├── PythonBindingXfSDxStereoBM.h
│   │   │       │   ├── xfSDxStereoBM.h
│   │   │       │   └── xfStereoBMCoreForVivadoHLS.h.in
│   │   │       └── src
│   │   │           ├── xfStereoBMCoreForVivadoHLS.cpp.in
│   │   │           └── xfStereoBM.cpp.in
│   │   ├── subtract
│   │   │   └── xfSDxKernel
│   │   │       ├── inc
│   │   │       │   ├── PythonBindingXfSDxSubtract.h
│   │   │       │   ├── xfSDxSubtract.h
│   │   │       │   └── xfSubtractCoreForVivadoHLS.h.in
│   │   │       └── src
│   │   │           ├── xfSubtractCoreForVivadoHLS.cpp.in
│   │   │           └── xfSubtract.cpp.in
│   │   └── threshold
│   │       └── xfSDxKernel
│   │           ├── inc
│   │           │   ├── PythonBindingXfSDxThreshold.h
│   │           │   ├── xfSDxThreshold.h
│   │           │   └── xfThresholdCoreForVivadoHLS.h.in
│   │           └── src
│   │               ├── xfThresholdCoreForVivadoHLS.cpp.in
│   │               └── xfThreshold.cpp.in
│   ├── frameworks
│   │   ├── cmakeModules
│   │   │   ├── FindVivadoHLS.cmake
│   │   │   ├── FindxfOpenCV.cmake
│   │   │   ├── rulesForSDxTargets.cmake
│   │   │   ├── rulesForSDxXfOpenCV.cmake
│   │   │   └── toolchain_sdx2018.2.cmake
│   │   └── utilities
│   │       ├── HRTimer
│   │       │   ├── CMakeLists.txt
│   │       │   ├── inc
│   │       │   │   └── HRTimer.h
│   │       │   └── src
│   │       │       └── HRTimer.cpp
│   │       ├── OpenCVPythonBindings
│   │       │   ├── cv_xilinx.cpp.in
│   │       │   ├── pycompat.hpp
│   │       │   ├── pyopencv_generated_include.h
│   │       │   └── xilinx_pyopencv_generated_ns_reg.h
│   │       ├── OpenCVUtils
│   │       │   ├── CMakeLists.txt
│   │       │   ├── inc
│   │       │   │   └── OpenCVUtils.h
│   │       │   └── src
│   │       │       └── OpenCVUtils.cpp
│   │       ├── SDxUtils
│   │       │   └── findAndInstallHwh.sh
│   │       └── xF
│   │           ├── Mat
│   │           │   ├── CMakeLists.txt
│   │           │   ├── inc
│   │           │   │   └── mat.hpp
│   │           │   └── src
│   │           │       └── mat.cpp
│   │           ├── PynqLib
│   │           │   ├── CMakeLists.txt
│   │           │   ├── inc
│   │           │   │   └── libxlnk_cma.h
│   │           │   └── src
│   │           │       ├── pynqlib.c
│   │           │       └── pynqlib.cpp
│   │           └── Utils
│   │               └── inc
│   │                   └── UtilsForXfOpenCV.h
│   ├── LICENSE
│   ├── overlays
│   │   ├── buildUnitOverlays.py
│   │   ├── cvXfUserSpecific
│   │   │   └── CMakeLists.txt
│   │   └── README.md
│   ├── pynq_cv
│   │   └── __init__.py
│   ├── README.md
│   └── setup.py
└── README.md
    
    
    
## Hardware and Software Environments
* Software Environments:
	* **OpenCV3.x** : https://www.opencv.org/
	* **xfOpenCV 2018.x** : https://github.com/Xilinx/xfopencv
	* **VisionWorks 1.6** : https://developer.nvidia.com/embedded/jetpack
 
* Hardware Platforms:
	* **FPGA board**: Xilinx Zynq UltraScale+ ZCU102/ZCU104, PYNQ-Z1 or Ultra96.
	* **GPU board**: NVIDIA Jetson TX1/ TX2.

 
## List of Vision Kernels

 
| Input Processing | Image Arithmatic | Filters       |  Image Analysis | Geometric Transforms|  Composite Kernels|
| -------------    | -------------    | ------------- | -------------   |    -------------    | --------------------    |
| combine          | AbsDiff          |  filter2D     |calcHist         | affine warp         | canny           |
| extract          | accumulate       |  box filter   |equalizeHist     |perspective warp     | fast       |      | 
| convertTo        |accumulate squared|  dilate   |integral image   | resize                  | harris                |
| cvtConvert       |accumulate weighted| erode        |mean std dev     | remap               | optical flow pyramid   |     
| table lookup     | add/subtract     |  median       |min/max loc      |                     | stereoBM     | 
|                  |  mulitply        | pyramidUp     |                 |                     |                    | 
|                  | threshold        | pyramidDown   |                 |                     |                 | 
|             | bitwise and,or,xor,not|               |                 |                     |                   | 
|                  | magnitude        |               |                 |                     |                    | 
|                  | phase            |               |                 |                     |                      | 
 


## Installation

To clone the repository with [PYNQ-ComputerVision](https://github.com/Xilinx/PYNQ-ComputerVision.git) submodules, open a terminal and execute:

```
git clone --recursive https://github.com/ISU-RCL/cvBench.git
```
## Build Test Codes 
The steps required to build and run unit tests is described in:

+ [GPU Implementation  (GPUTest)](GPUTests/README.md)  
+ [FPGA Implementation (PLTests)](FPGATests/README.md)

## Results Summary

These tables show the **energy/frame consumpsion (in mJ/f)** of vision kernels on CPU, GPU and FPGA.  
 

 **(1) Input Processing Kernels**
 
| Kernel/platform | combine | extract | convertTo| depth Convert |  
| ---------	| ------  | ------  | -------  	|    ------  |      
| ARM-57 CPU	|   3.1      |  2.9	    |	 2.4       |     4.5       |          
| GPU           |	  |  	    |	     	|            |            
| FPGA      	|  	  |	    |           |            | 		    
 

 **(2) Image Arithmatic Kernels**

| Kernel/platform		| AbsDiff | accumulate | accumulate squared	| accumulate weighted | add/subtract | mulitply| threshold | bitwise and,or,xor,not| magnitude | phase|
| ---------					| ------  | ------  | -------  	|    ------  |     ------   |     ------  |     ------   |    ------  |     ------   |    ------   |  
|    ARM-57 CPU			|         |  	     	|	        	|            |              |          |  	     	|	        	|            |              |  
|    GPU		|         |  	     	|	        	|            |              |          |  	     	|	        	|            |              |  
|    FPGA			|         |  	     	|	        	|            |              |          |  	     	|	        	|            |              |  

 **(3) Filters Kernels**

| Kernel/platform		| filter2D | box filter | dilate	| erode | median | pyramidUp| pyramidDown|
| ---------					| ------  | ------  | -------  	|    ------  |     ------   |   ------  |     ------   |  
|    ARM-57 CPU			|         |  	     	|	        	|            |              |           |              | 
|    GPU      			|	  			|  	     	|	     			|           |              |            |              | 
|    FPGA      			|	  			|  	     	|	     			|            |              |           |              | 


 **(4) Image Analysis Kernels**

| Kernel/platform		| calcHist | equalizeHist | integral image	| mean std dev | min/max loc |
| ---------					| ------  | ------  | -------  	|    ------  |     ------   |  
|    ARM-57 CPU			|         |  	     	|	        	|            |              | 
|    GPU      			|	  			|  	     	|	     			|            |              | 
|    FPGA      			|	  			|  	     	|	     			|            |              | 
 
 **(5) Geometric Transforms Kernels**

| Kernel/platform		| affine warp  | perspective warp  | resize 	| remap |  
| ---------					| ------  | ------  | -------  	|    ------  |     
|    ARM-57 CPU			|         |  	     	|	        	|            |          
|    GPU      			|	  			|  	     	|	          |              | 
|    FPGA      			|	  			|  	     	|	          |              | 


 **(6) Composite Kernels**

| Kernel/platform		| canny | fast | harris 	| optical flow pyramid |  stereoBM
| ---------					| ------  | ------  | -------  	|    ------  |   ------  |     
|    ARM-57 CPU			|         |  	     	|	        	|            |   |          
|    GPU      			|	  			|  	     	|	          |              |  |  
|    FPGA      			|	  			|  	     	|	          |              |  |  


  
## Reference 

```
@inproceedings{EnergyEfficiencyFCCM2019,
  title={Analyzing the Energy-Efficiency of Vision Kernels on Embedded CPU, GPU and FPGA Platforms},
  author={Qasaimeh, Murad and Kristof, Denolf and Jack, Lo and Kees, Vissers and Zambreno, Joseph and Jones, Phillip H},
  booktitle={2019 IEEE 27th Annual International Symposium on Field-Programmable Custom Computing Machines (FCCM)},
  year={2019},
  organization={IEEE}
}
```
## License
The source for this project is licensed under the [3-Clause BSD License](LICENSE)

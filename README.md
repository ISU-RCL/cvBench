## Benchmarking Analysis of Vision Kernels on Embedded CPU, GPU and FPGA:


<p align="justify">

This repository contains benchmark framework for measuring and comparing energy efficiency of different vision kernels on embedded platforms. It aims to provide computer vision community an easy tool to analyze the performance of vision kernels on different hardware architectures and aids with determining which hardware architecture is most suitable for different kind of vision applications.

</p>

## Table of contents 
<!--ts-->
* [Repository Structure](#Repository-Structure) 
* [Hardware and Software Environments](#Hardware-and-Software-Environments)
* [List of Vision Kernels](#List-of-Vision-Kernels)
* [Installation](#Installation) 
* [Build Test Codes](#Build-Test-Codes) 
* [Results Summary](#Results-Summary) 
* [Reference](#reference)
* [License](#license) 
<!--te-->
   
## Repository Structure

This repository consists of:
 ```   
.
├── FPGATests
│   └── README.md
├── GPUTests
│   ├── Geometric Transforms 
│   ├── Image Analysis  
│   ├── Image Arithmatic 
│   ├── Image Features 
│   ├── Image Filters 
│   ├── Input Processing 
│   ├── Optical Flow & Depth 
│   └── README.md 
├── PYNQ-ComputerVision
│   ├── applicationCode 
│   │   ├── overlayTests 
│   │   └── unitTests
│   ├── boards
│   │   ├── Pynq-Z1
│   │   ├── Pynq-Z2
│   │   ├── Ultra96
│   │   └── ZCU104
│   ├── components 
│   └── frameworks
│       ├── cmakeModules 
│       └── utilities    
└── README.md
``` 

## Hardware and Software Environments
* Software Environments:
	* **OpenCV3.x** : https://www.opencv.org/
	* **xfOpenCV 2018.x** : https://github.com/Xilinx/xfopencv
	* **VisionWorks 1.6** : https://developer.nvidia.com/embedded/jetpack
 
* Hardware Platforms:
	* **FPGA board**: [Xilinx Zynq ZCU102/ZCU104](https://www.xilinx.com/products/boards-and-kits/ek-u1-zcu102-g.html), [PYNQ-Z1/2](http://www.pynq.io/board) or [Ultra96](http://zedboard.org/product/ultra96).
	* **GPU board**: [NVIDIA Jetson TX1/ TX2](https://developer.nvidia.com/embedded/buy/jetson-tx2).

 
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

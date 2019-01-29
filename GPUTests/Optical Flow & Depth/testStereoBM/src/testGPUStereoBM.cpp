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
*	(c) Copyright 2019 Iowa State University.
*	All rights reserved.
*	
*	Author: Murad Qasaimeh <qasaimeh@iastate.edu>
*	Date:   2018/03/27
*****************************************************************************/

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include <opencv2/highgui/highgui.hpp>
#include "OpenCVUtils.h" 
#include "opencv2/core/cuda.hpp" 
#include "ivx.hpp"       

#include <HRTimer.h> 
using namespace cv;
void cv_StereoPipeline(cv::Mat &left, cv::Mat &right, cv::Mat &disparitySW);

int main(int argc, char** argv)
{
	HRTimer timer;
	
	try {	
		const cv::String keys =
			"{help h usage ? |                 | print this message                   }"
			"{@LeftImage     |                 | input image 	                      }"
			"{@RightImage    |                 | input image 	                      }"
			"{goldenFile gf  |                 | golden output image (from SW)        }"
			"{outFile of     |                 | output image (from PL)               }"
			"{display d      |                 | diplay result with imshow            }"
			"{iterations n   | 1               | number of iterations to measure time }"
			;

		cv::CommandLineParser parser(argc, argv, keys.c_str());
		if (parser.has("help") || argc < 2)
		{
			parser.printMessage();
			std::cout << "\nhit enter to quit...";
			std::cin.get();
			return 0;
		}

		cv::String filenameLeft  = parser.get<cv::String>(0);
		cv::String filenameRight = parser.get<cv::String>(1);

		cv::String filenameSW; bool writeSWResult = false;
		if (parser.has("goldenFile")) {
			filenameSW = parser.get<cv::String>("goldenFile");
			writeSWResult = true;
		}

		cv::String filenamePL; bool writePLResult = false;
		if (parser.has("outFile")) {
			filenamePL = parser.get<cv::String>("outFile");
			writePLResult = true;
		}

		bool imShowOn = parser.has("display");
		unsigned int numberOfIterations;
		if (parser.has("iterations"))
			numberOfIterations = parser.get<unsigned int>("iterations");

		if (!parser.check())
		{
			parser.printErrors();
			return(-1);
		}

		// Initialize
    	cv::Mat srcLeft, srcRight, srcLeftGray, srcRightGray, disparitySW;

   		initializeDualImageTest(filenameLeft, filenameRight, srcLeft, srcRight); 

		//convert to grayscale
   		cvtColor(srcLeft,  srcLeftGray,  CV_BGR2GRAY, 1);
   		cvtColor(srcRight, srcRightGray, CV_BGR2GRAY, 1);

		int width  = srcLeft.size().width;
		int height = srcLeft.size().height;

  	 	int numberOfDisparitiesD=16;
		int blockSizeD= 9;

		// Apply OpenCV stereoBM
	    Ptr<StereoBM> sbm = cv::StereoBM::create(numberOfDisparitiesD, blockSizeD);
	    sbm->setPreFilterCap(31);	
 	   	sbm->setUniquenessRatio(15);
  		sbm->setTextureThreshold(20);
    	sbm->setMinDisparity(0);   		 			
		
		std::cout << "running on CPU" << std::endl;

		//warming up
       	sbm->compute(srcLeftGray, srcRightGray, disparitySW);

    	timer.StartTimer();
    	for (int i = 0; i < numberOfIterations; i++){
       		sbm->compute(srcLeftGray, srcRightGray, disparitySW);
    	}
    	timer.StopTimer();
 
		std::cout << "Elapsed time over " << numberOfIterations << "SW call(s): " << timer.GetElapsedUs() << " us or " << (float)timer.GetElapsedUs() / (float)numberOfIterations << "us per frame" << std::endl;

		//convert depth to 16 bits
		cv::Mat tmpMatSW;
		disparitySW.convertTo(tmpMatSW, CV_16U);
			 
	 
		
		/*-------------------------- CUDA ---------------------------------------------*/
		bool ranCuda = false;
		cv::Mat dstCudaOnHost, tmpMatCuda;
		
#ifdef WITH_CUDA
		
		std::cout << "CUDA gpu device count: " << cv::cuda::getCudaEnabledDeviceCount() << std::endl;
		if (cv::cuda::getCudaEnabledDeviceCount() < 1) {
			std::cout << "no Cuda devices available" << std::endl;
		}
		else
		{		
			cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
	
			float cameraMAL[9]= {933.173, 0.0, 663.451, 0.0, 933.173, 377.015, 0.0, 0.0, 1.0};
			float cameraMAR[9] ={933.467, 0.0, 678.297, 0.0, 933.467, 359.623, 0.0, 0.0, 1.0};
			 
			float distCL[5] = {-0.169398, 0.02273290, 0.0, 0.0, 0.0};
			float distCR[5] = {-0.170581, 0.02494440, 0.0, 0.0, 0.0};

			float irAL[9] = {0.0011976323, -0.0000000019,-0.8153011732,
							 0.0000000007,  0.0011976994,-0.4422348617,
							 0.0000126839,  0.0000001064, 0.9913820905}; 

			float irAR[9] = {0.0011976994, 0.0000000000,-0.8047567905,
							-0.0000000000, 0.0011976994,-0.4420566166,
							-0.0000000000,-0.0000001064, 1.0000392898};
			 
			cv::Mat cameraMatrixL(3, 3, cv::DataType<float>::type);
			cv::Mat cameraMatrixR(3, 3, cv::DataType<float>::type);
			cv::Mat rectifyMatrixL(3, 3, cv::DataType<float>::type);
			cv::Mat rectifyMatrixR(3, 3, cv::DataType<float>::type);
			cv::Mat distMatrixL(5, 1, cv::DataType<float>::type);  
			cv::Mat distMatrixR(5, 1, cv::DataType<float>::type);   
	
			// copy camera params
			for(int i=0; i<3; i++) {
				for(int j=0; j<3; j++) 
				{
					cameraMatrixL.at<float>(i, j)=cameraMAL[i*3+j];			
					cameraMatrixR.at<float>(i, j)=cameraMAR[i*3+j]; 
					rectifyMatrixL.at<float>(i, j)=irAL[i*3+j];			
					rectifyMatrixR.at<float>(i, j)=irAR[i*3+j]; 			  
				}
			}
			// copy distortion coefficients
			for(int i=0; i<5; i++) {
				distMatrixL.at<float>(i)= distCL[i];
				distMatrixR.at<float>(i)= distCR[i];
			}  

			cv::Mat mapxL, mapyL, mapxR, mapyR;

			timer.StartTimer();
			cv::initUndistortRectifyMap(cameraMatrixL, distMatrixL, cv::Mat(), cv::Mat(), Size(1920,1080),CV_32FC1, mapxL, mapyL); 	 	
			cv::initUndistortRectifyMap(cameraMatrixR, distMatrixR, cv::Mat(), cv::Mat(), Size(1920,1080),CV_32FC1, mapxR, mapyR);   
		 	timer.StopTimer(); 
			std::cout << "initUndistortRectifyMap=  " << timer.GetElapsedUs() << " us "   << std::endl;
			
 
			ivx::Context context = ivx::Context::create();
			//copy cv::Mat to vx_image   
		    ivx::Image ivxsrcLeftGray  = ivx::Image::create(context, width, height,  VX_DF_IMAGE_U8);
		    ivx::Image ivxsrcRightGray = ivx::Image::create(context, width, height,  VX_DF_IMAGE_U8);
		    ivxsrcLeftGray.copyFrom(0, srcLeftGray);  
		    ivxsrcRightGray.copyFrom(0, srcRightGray);  

			//dst image
	 		ivx::Image ivxsrcLeftRectified = ivx::Image::create(context, width, height,  VX_DF_IMAGE_U8); 
	 		ivx::Image ivxsrcRightRectified = ivx::Image::create(context, width, height,  VX_DF_IMAGE_U8); 

	 		vx_remap mapivxL = vxCreateRemap(context, width, height, width, height);
	 		vx_remap mapivxR = vxCreateRemap(context, width, height, width, height); 

			for( int j = 0; j < height; j++ )
			{
				for( int i = 0; i < width; i++ )
				{ 
	 				vxSetRemapPoint(mapivxL, i, j, mapxL.at<uchar>(j, i), mapyL.at<uchar>(j, i));
	 				vxSetRemapPoint(mapivxR, i, j, mapxR.at<uchar>(j, i), mapyR.at<uchar>(j, i));
				}
			}

			//create graph
 			ivx::Graph graph = ivx::Graph::create(context);  
			ivx::Node::create(graph, VX_KERNEL_REMAP, ivxsrcLeftGray, mapivxL, ivx::Scalar::create<VX_TYPE_ENUM>(context, VX_INTERPOLATION_TYPE_BILINEAR), ivxsrcLeftRectified);  
			ivx::Node::create(graph, VX_KERNEL_REMAP, ivxsrcRightGray, mapivxR, ivx::Scalar::create<VX_TYPE_ENUM>(context, VX_INTERPOLATION_TYPE_BILINEAR), ivxsrcRightRectified);  
		 
			std::cout << "running OpenVX" << std::endl;			
		
			timer.StartTimer();
			graph.process();   
			timer.StopTimer();
 
			std::cout << "Remap OpenVX= "<< timer.GetElapsedUs() <<" us" << std::endl;
			
			cv::Mat LeftRect, RightRect;

        	ivxsrcLeftRectified.copyTo(0, LeftRect);
        	ivxsrcRightRectified.copyTo(0, RightRect);

			//cv::cuda
			cv::cuda::GpuMat leftRectifiedCuda, rightRectifiedCuda, disparityCuda;
			
 			Ptr<cv::StereoBM> sbmCuda = cv::cuda::createStereoBM(numberOfDisparitiesD, blockSizeD);
	    	sbmCuda->setPreFilterCap(31);	
 	   		sbmCuda->setUniquenessRatio(15);
  			sbmCuda->setTextureThreshold(20);
    		sbmCuda->setMinDisparity(0);  

			leftRectifiedCuda.upload(LeftRect);
			rightRectifiedCuda.upload(RightRect);

			timer.StartTimer();
			for (int i = 0; i < numberOfIterations; i++) { 
       			sbmCuda->compute(leftRectifiedCuda, rightRectifiedCuda, disparityCuda);
			}
			timer.StopTimer();

			
			disparityCuda.download(dstCudaOnHost);
			std::cout << "StereoBN= " << timer.GetElapsedUs() << " us"  << std::endl;
			ranCuda = true;

			//convert depth to 16 bits 
			dstCudaOnHost.convertTo(tmpMatCuda, CV_16U);
			std::cout<<"tmpMatCuda"<<tmpMatCuda.size()<<std::endl;
		}
#endif

		/*-------------------------- OPENVX ---------------------------------------------*/
		bool ranOpenvx = false;
		cv::Mat dstOpenvx; 

#ifdef WITH_OPENVX
 
			std::cout << "VisionWorks's StereoBM is not avaiable" << std::endl;
#endif

		/*------------- end of measurements ----------------------*/
		
		// compare results
		int numberOfDifferences = 0;
		double errorPerPixel = 0; 
		if (ranCuda)
		{
			std::cout << "comparing CUDA versus CPU" << std::endl;
			imageCompare(dstCudaOnHost, tmpMatSW, numberOfDifferences, errorPerPixel, true, false);
			std::cout << "number of differences: " << numberOfDifferences << " average error per pixel: " << errorPerPixel << std::endl;
		}
		if (ranOpenvx)
		{
			std::cout << "comparing Openvx versus CPU" << std::endl;
			imageCompare(dstOpenvx, tmpMatSW, numberOfDifferences, errorPerPixel, true, false);
			std::cout << "number of differences: " << numberOfDifferences << " average error per pixel: " << errorPerPixel << std::endl;
		}
		
		//show results
		if (imShowOn) {
       		imshow("Input left", srcLeft);
        	imshow("Input right", srcRight);

			double minVal; double maxVal;
        	minMaxLoc(tmpMatSW, &minVal, &maxVal);
        
 			cv::Mat tmpShowDisparitySW, showDisparitySW;
        	tmpMatSW.convertTo(tmpShowDisparitySW,CV_8U, 255.0/maxVal); // use maxVal for nicer colors instead of (numberOfDisparitiesD*16.0) disparity in 16U is 12.4 format, so we need *16
        	applyColorMap(tmpShowDisparitySW, showDisparitySW, cv::COLORMAP_JET);        
        	imshow("Processed (SW)", showDisparitySW);

 
			if (ranCuda)
			{				
				double minValCuda, maxValCuda;
		    	minMaxLoc(tmpMatCuda, &minValCuda, &maxValCuda);
		    
	 			cv::Mat tmpShowDisparityCuda, showDisparityCuda;
		    	tmpMatCuda.convertTo(tmpShowDisparityCuda,CV_8U, 255.0/maxValCuda); // use maxVal for nicer colors instead of (numberOfDisparitiesD*16.0) disparity in 16U is 12.4 format, so we need *16
		    	applyColorMap(tmpShowDisparityCuda, showDisparityCuda, cv::COLORMAP_JET);        
		    	imshow("Processed (CUDA)", showDisparityCuda);
			} 
			if (ranOpenvx)
				cv::imshow("Processed (Openvx)", dstOpenvx);
			cv::waitKey(0);
		}
	}
	catch (std::exception &e)
	{
		const char* errorMessage = e.what();
		std::cerr << "Exception caught: " << errorMessage << std::endl;
		std::cout << "\nhit enter to quit...";
		std::cin.get();
		exit(-1);
	}
	catch (const char *e) {
		std::cerr << "Exception caught: " << e << std::endl;
		std::cout << "\nhit enter to quit...";
		std::cin.get();
		exit(-1);
	}

	std::cout << "\nhit enter to quit...";
	std::cin.get();
	return 0;
}

void cv_StereoPipeline(cv::Mat &left, cv::Mat &right, cv::Mat &disparitySW)
{  
	float cameraMAL[9]= {933.173, 0.0, 663.451, 0.0, 933.173, 377.015, 0.0, 0.0, 1.0};
	float cameraMAR[9] ={933.467, 0.0, 678.297, 0.0, 933.467, 359.623, 0.0, 0.0, 1.0};
	 
	float distCL[5] = {-0.169398, 0.02273290, 0.0, 0.0, 0.0};
	float distCR[5] = {-0.170581, 0.02494440, 0.0, 0.0, 0.0};

	float irAL[9] = {0.0011976323, -0.0000000019,-0.8153011732,
					 0.0000000007,  0.0011976994,-0.4422348617,
					 0.0000126839,  0.0000001064, 0.9913820905}; 

	float irAR[9] = {0.0011976994, 0.0000000000,-0.8047567905,
					-0.0000000000, 0.0011976994,-0.4420566166,
					-0.0000000000,-0.0000001064, 1.0000392898};
	 
	cv::Mat cameraMatrixL(3, 3, cv::DataType<float>::type);
	cv::Mat cameraMatrixR(3, 3, cv::DataType<float>::type);
	cv::Mat rectifyMatrixL(3, 3, cv::DataType<float>::type);
	cv::Mat rectifyMatrixR(3, 3, cv::DataType<float>::type);
	cv::Mat distMatrixL(5, 1, cv::DataType<float>::type);  
	cv::Mat distMatrixR(5, 1, cv::DataType<float>::type);   
	
	// copy camera params
	for(int i=0; i<3; i++) {
		for(int j=0; j<3; j++) 
		{
			cameraMatrixL.at<float>(i, j)=cameraMAL[i*3+j];			
			cameraMatrixR.at<float>(i, j)=cameraMAR[i*3+j]; 
			rectifyMatrixL.at<float>(i, j)=irAL[i*3+j];			
			rectifyMatrixR.at<float>(i, j)=irAR[i*3+j]; 			  
		}
	}
	// copy distortion coefficients
	for(int i=0; i<5; i++) {
		distMatrixL.at<float>(i)= distCL[i];
		distMatrixR.at<float>(i)= distCR[i];
	}  
	
	cv::Mat mapxL, mapyL, mapxR, mapyR;
	cv::Mat leftRectified, rightRectified;
	cv::initUndistortRectifyMap(cameraMatrixL, distMatrixL, cv::Mat(), cv::Mat(), Size(1280,720),CV_32FC1, mapxL, mapyL);  
 	remap(left, leftRectified, mapxL, mapyL, cv::INTER_LINEAR); 
 	
	cv::initUndistortRectifyMap(cameraMatrixR, distMatrixR, cv::Mat(), cv::Mat(), Size(1280,720),CV_32FC1, mapxR, mapyR);  
 	remap(right, rightRectified, mapxR, mapyR, cv::INTER_LINEAR);  	
 
 	Ptr<cv::StereoBM> sbm = cv::StereoBM::create(64, 19);
	sbm->setPreFilterCap(31);
	sbm->setUniquenessRatio(15);
	sbm->setTextureThreshold(20);
	sbm->setMinDisparity(0);
	
    sbm->compute(leftRectified, rightRectified, disparitySW);
	 		

} 

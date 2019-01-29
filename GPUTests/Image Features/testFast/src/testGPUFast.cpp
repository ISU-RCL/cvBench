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

int main(int argc, char** argv)
{
	HRTimer timer;
	
	try {	
		const cv::String keys =
			"{help h usage ? |                 | print this message                   }"
			"{@image         |                 | input image 	                      }"
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

		cv::String filenameIn = parser.get<cv::String>(0);

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
		cv::Mat srcIn, dstSW;
		initializeSingleGrayImageTest(filenameIn, srcIn);

		int width = srcIn.size().width;
		int height = srcIn.size().height;
  

		std::cout << "running on CPU" << std::endl;
		double threshold= 30;
		bool nonmaxSuppression=true;

		cv::Ptr<cv::FastFeatureDetector> fastDetector = cv::FastFeatureDetector::create(threshold, nonmaxSuppression);
		std::vector<cv::KeyPoint> keypointsSW;
		
		//warming up
		fastDetector->detect(srcIn, keypointsSW); 						
		timer.StartTimer();
		for (int i = 0; i < numberOfIterations; i++) 
		{
			fastDetector->detect(srcIn, keypointsSW);
		}
		timer.StopTimer(); 						
		std::cout << "Elapsed time over " << numberOfIterations << "SW call(s): " << timer.GetElapsedUs() << " us or " << (float)timer.GetElapsedUs() / (float)numberOfIterations << "us per frame" << std::endl;

		
		/*-------------------------- CUDA ---------------------------------------------*/
		bool ranCuda = false;
		cv::Mat dstCudaOnHost;
		std::vector<cv::KeyPoint> gpuKeypoints;

#ifdef WITH_CUDA
		
		std::cout << "CUDA gpu device count: " << cv::cuda::getCudaEnabledDeviceCount() << std::endl;
		if (cv::cuda::getCudaEnabledDeviceCount() < 1) {
			std::cout << "no Cuda devices available" << std::endl;
		}
		else
		{	
			cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
	
			cv::cuda::GpuMat srcCuda, dstCuda;
			srcCuda.upload(srcIn); 

			std::cout << "running CUDA" << std::endl;	 
			cv::Ptr<cv::cuda::FastFeatureDetector> gpuFastDetector = cv::cuda::FastFeatureDetector::create(threshold, nonmaxSuppression);			

			//warming up 
			gpuFastDetector->detect(srcCuda, gpuKeypoints); 						
			timer.StartTimer();
			for (int i = 0; i < numberOfIterations; i++) 
			{
				gpuFastDetector->detect(srcCuda, gpuKeypoints);		
			}
			timer.StopTimer(); 						 
			std::cout << "Elapsed time over " << numberOfIterations << "CUDA call(s): " << timer.GetElapsedUs() << " us or " << (float)timer.GetElapsedUs() / (float)numberOfIterations << "us per frame" << std::endl;		

			ranCuda = true;
		}
#endif

		/*-------------------------- OPENVX ---------------------------------------------*/
		bool ranOpenvx = false;
		cv::Mat dstOpenvx; 
		std::vector<cv::KeyPoint> openVXKeypoints; 
#ifdef WITH_OPENVX
		//create context
		ivx::Context context = ivx::Context::create();

		//copy cv::Mat to vx_image   
        ivx::Image ivxImage = ivx::Image::create(context, width, height, VX_DF_IMAGE_U8);
        ivxImage.copyFrom(0, srcIn); 

  		ivx::Scalar thresholdVX = ivx::Scalar::create<VX_TYPE_FLOAT32>(context, threshold);  		
		ivx::Array corners      = ivx::Array::create(context, VX_TYPE_KEYPOINT,1000);
  		ivx::Scalar num_corners = ivx::Scalar::create<VX_TYPE_SIZE>(context, 1000);  	  		
		ivx::Scalar nonmaxSuppressionVX = ivx::Scalar::create<VX_TYPE_BOOL >(context, vx_true_e);   
		//create graph
 		ivx::Graph graph = ivx::Graph::create(context); 
		 
		ivx::Node::create(graph, VX_KERNEL_FAST_CORNERS, ivxImage,thresholdVX,nonmaxSuppressionVX,corners, num_corners);
		 
		std::cout << "running OpenVX" << std::endl;			
		
		//warming up
 		graph.process();  							
		timer.StartTimer();
		for (int i = 0; i < numberOfIterations; i++){
 			graph.process(); 
		}
		timer.StopTimer(); 							
        //copy vx_image to cv::Mat
        size_t nPoints = corners.itemCount();
        std::vector<vx_keypoint_t> vxCorners;
        corners.copyTo(vxCorners); 
        for(size_t i = 0; i < nPoints; i++)
        {
            vx_keypoint_t kp = vxCorners[i]; 
            openVXKeypoints.push_back(cv::KeyPoint((float)kp.x, (float)kp.y, 7.f, -1, kp.strength)); 
		} 

		std::cout << "Elapsed time over " << numberOfIterations << " OpenVX call(s): " << timer.GetElapsedUs() << " us or " << (float)timer.GetElapsedUs() / (float)numberOfIterations << "us per frame" << std::endl;

		ranOpenvx = true;
 
#endif

		/*------------- end of measurements ----------------------*/
 		
		//show results
		if (imShowOn) {
			cv::imshow("Input image", srcIn);
			
			cv::drawKeypoints(srcIn, keypointsSW, dstSW, cv::Scalar(0,0,255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
			cv::imshow("Processed (SW)", dstSW);
 
			if (ranCuda)
			{
				cv::drawKeypoints(srcIn, gpuKeypoints, dstCudaOnHost, cv::Scalar(0,255,255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
				cv::imshow("Processed (CUDA)", dstCudaOnHost);
			}	
			if (ranOpenvx)
			{	
				cv::drawKeypoints(srcIn, openVXKeypoints, dstOpenvx, cv::Scalar(0,255,255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
				cv::imshow("Processed (Openvx)", dstOpenvx);
			}
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
 

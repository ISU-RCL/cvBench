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

using namespace std;
using namespace cv;
using namespace cv::cuda;


static void download(const GpuMat& d_mat, vector<Point2f>& vec)
{
    vec.resize(d_mat.cols);
    Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
    d_mat.download(mat);
}

static void download(const GpuMat& d_mat, vector<uchar>& vec)
{
    vec.resize(d_mat.cols);
    Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
    d_mat.download(mat);
}


int main(int argc, char** argv)
{
	HRTimer timer;
	
	try {	
		const cv::String keys =
			"{help h usage ? |                 | print this message                   }"
			"{@PrevImage     |                 | input image 	                      }"
			"{@CurrImage    |                 | input image 	                      }"
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

		cv::String filenamePrev  = parser.get<cv::String>(0);
		cv::String filenameCurr = parser.get<cv::String>(1);

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
    	cv::Mat prevImg, currImg, prevImgGray, currImgGray;

   		initializeDualImageTest(filenamePrev, filenameCurr, prevImg, currImg); 
 
        cv::cvtColor(prevImg, prevImgGray, CV_RGB2GRAY);
        cv::cvtColor(currImg, currImgGray, CV_RGB2GRAY);

 
        std::vector<cv::Point2f> prevCorners;
        std::vector<cv::Point2f> currCorners;

        cv::goodFeaturesToTrack(prevImgGray, prevCorners, 200, 0.5, 5.0);
        cv::goodFeaturesToTrack(currImgGray, currCorners, 200, 0.5, 5.0);
       
		cv::cornerSubPix(prevImgGray, prevCorners, cv::Size(21, 21), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 100, 0.01));
        cv::cornerSubPix(currImgGray, currCorners, cv::Size(21, 21), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 100, 0.01));

        std::vector<uchar> featuresFound;
        std::vector<float> featuresErrors;

        cv::calcOpticalFlowPyrLK( prevImg, currImg, prevCorners, currCorners, featuresFound, featuresErrors);

        for (int i = 0; i < featuresFound.size(); i++) {
            cv::Point p1 = cv::Point((int) prevCorners[i].x, (int) prevCorners[i].y);
            cv::Point p2 = cv::Point((int) currCorners[i].x, (int) currCorners[i].y);
            cv::line(prevImg, p1, p2, cv::Scalar(0, 0, 255), 2);
        }
		 			
		 
		std::cout << "running on CPU" << std::endl;

		//warming up
        cv::calcOpticalFlowPyrLK( prevImg, currImg, prevCorners, currCorners, featuresFound, featuresErrors);

    	timer.StartTimer();
    	for (int i = 0; i < numberOfIterations; i++){
        	cv::calcOpticalFlowPyrLK( prevImg, currImg, prevCorners, currCorners, featuresFound, featuresErrors);
    	}
    	timer.StopTimer();
 
		std::cout << "Elapsed time over " << numberOfIterations << "SW call(s): " << timer.GetElapsedUs() << " us or " << (float)timer.GetElapsedUs() / (float)numberOfIterations << "us per frame" << std::endl;
 
 
		
		/*-------------------------- CUDA ---------------------------------------------*/
		bool ranCuda = false;
		cv::Mat dstCudaOnHost, tmpMatCuda;
		cv::cvtColor(prevImg, prevImgGray, CV_RGB2GRAY);
        cv::cvtColor(currImg, currImgGray, CV_RGB2GRAY);

		int width = prevImgGray.size().width;
		int height = prevImgGray.size().height; 
		std::vector<cv::KeyPoint> openVXKeypoints; 
#ifdef WITH_CUDA
		
		std::cout << "CUDA gpu device count: " << cv::cuda::getCudaEnabledDeviceCount() << std::endl;
		if (cv::cuda::getCudaEnabledDeviceCount() < 1) {
			std::cout << "no Cuda devices available" << std::endl;
		}
		else
		{		
 			// goodFeaturesToTrack
   			cv::cuda::GpuMat d_frame0Gray(prevImgGray);
    		cv::cuda::GpuMat d_prevPts; 

    		Ptr<cuda::CornersDetector> detector = cuda::createGoodFeaturesToTrackDetector(d_frame0Gray.type(), 1000  , 0.01, 0);
    		detector->detect(d_frame0Gray, d_prevPts);
   			 
			// Sparse
    		Ptr<cuda::SparsePyrLKOpticalFlow> d_pyrLK = cuda::SparsePyrLKOpticalFlow::create(Size(21, 21), 3, 30);
			cv::cuda::GpuMat d_frame0(prevImgGray);
			cv::cuda::GpuMat d_frame1(currImgGray);
			cv::cuda::GpuMat d_frame1Gray(currImgGray);
			cv::cuda::GpuMat d_nextPts;
			cv::cuda::GpuMat d_status;

 
    		timer.StartTimer();
    		for (int i = 0; i < numberOfIterations; i++){
				d_pyrLK->calc(true ? d_frame0Gray : d_frame0, true ? d_frame1Gray : d_frame1, d_prevPts, d_nextPts, d_status);
			}
			timer.StopTimer();
			std::cout << "Elapsed time over " << numberOfIterations << "CUDA call(s): " << timer.GetElapsedUs() << " us or " << (float)timer.GetElapsedUs() / (float)numberOfIterations << "us per frame" << std::endl;
 
			
			std::vector<Point2f> prevPts(d_prevPts.cols);
			download(d_prevPts, prevPts);

			std::vector<Point2f> nextPts(d_nextPts.cols);
			download(d_nextPts, nextPts);

			std::vector<uchar> status(d_status.cols);
			download(d_status, status);


		}
#endif

		/*-------------------------- OPENVX ---------------------------------------------*/
		bool ranOpenvx = false;
		cv::Mat dstOpenvx; 

#ifdef WITH_OPENVX
 
		//create context
//create context
		ivx::Context context = ivx::Context::create();

		//copy cv::Mat to vx_image   
        ivx::Image ivxImagePrev = ivx::Image::create(context, width, height, VX_DF_IMAGE_U8);
        ivx::Image ivxImageNext = ivx::Image::create(context, width, height, VX_DF_IMAGE_U8);
        ivxImagePrev.copyFrom(0, prevImgGray); 
        ivxImageNext.copyFrom(0, currImgGray); 
 
		//Harris Params
  		ivx::Scalar thresholdVX = ivx::Scalar::create< VX_TYPE_FLOAT32>(context, 0.0009);  	
  		ivx::Scalar min_distance = ivx::Scalar::create< VX_TYPE_FLOAT32>(context, 0);  	
  		ivx::Scalar sensitivity = ivx::Scalar::create< VX_TYPE_FLOAT32>(context, 0.000014);  		 	
  		ivx::Scalar gradient_size = ivx::Scalar::create<VX_TYPE_INT32>(context, 3);  		 	
  		ivx::Scalar block_size = ivx::Scalar::create<VX_TYPE_INT32>(context, 3);  	 
  		ivx::Scalar num_corners = ivx::Scalar::create<VX_TYPE_SIZE>(context, 1000);  	

		ivx::Array prev_features = ivx::Array::create(context, VX_TYPE_KEYPOINT, 1000);
		ivx::Array new_features = ivx::Array::create(context, VX_TYPE_KEYPOINT, 1000);   		 		
		ivx::Array est_features = ivx::Array::create(context, VX_TYPE_KEYPOINT, 1000);   		 

		//create graph
 		ivx::Graph graph = ivx::Graph::create(context); 
		 
		ivx::Node::create(graph, VX_KERNEL_HARRIS_CORNERS, ivxImagePrev, thresholdVX, min_distance, sensitivity, gradient_size,block_size, prev_features, num_corners);
		 

 		std::cout << "prev_features.itemCount() "<< prev_features.itemCount()<< std::endl;	

 		ivx::Pyramid pyramid_next = ivx::Pyramid::create(context, 4, VX_SCALE_PYRAMID_HALF, width, height, VX_DF_IMAGE_U8);
 		ivx::Pyramid pyramid_prev  = ivx::Pyramid::create(context, 4, VX_SCALE_PYRAMID_HALF, width, height, VX_DF_IMAGE_U8); 

		ivx::Node::create(graph, VX_KERNEL_GAUSSIAN_PYRAMID, ivxImagePrev, pyramid_prev);  
		ivx::Node::create(graph, VX_KERNEL_GAUSSIAN_PYRAMID, ivxImageNext, pyramid_next);  

		//minEigThreshold is fixed to 0.0001f
        ivx::Scalar termination = ivx::Scalar::create<VX_TYPE_ENUM>(context, VX_TERM_CRITERIA_BOTH);
        ivx::Scalar epsilon = ivx::Scalar::create<VX_TYPE_FLOAT32>(context, 0.01);
        ivx::Scalar numIterations = ivx::Scalar::create<VX_TYPE_UINT32>(context, 100);
        ivx::Scalar useInitial = ivx::Scalar::create<VX_TYPE_BOOL>(context, (vx_bool)(vx_true_e)); 
        ivx::Scalar windowSize = ivx::Scalar::create<VX_TYPE_SIZE>(context, (vx_size)32);

        ivx::Node::create(graph, VX_KERNEL_OPTICAL_FLOW_PYR_LK, pyramid_prev, pyramid_next, prev_features, prev_features, new_features, termination, epsilon, numIterations, useInitial, windowSize);
 		

		std::cout << "running OpenVX" << std::endl;			
		
		//warming up
 		graph.process();  				
		timer.StartTimer();
		for (int i = 0; i < numberOfIterations; i++){
 			graph.process(); 
		}
		timer.StopTimer(); 					 
        //copy vx_image to cv::Mat
        size_t nPrevPoints = prev_features.itemCount();
        size_t nNextPoints = new_features.itemCount();
        std::vector<vx_keypoint_t> vxPrev_features;
        std::vector<vx_keypoint_t> vxNext_features;
 		std::cout << nPrevPoints<<" "<< nNextPoints<< std::endl;			
        prev_features.copyTo(vxPrev_features);      
		new_features.copyTo(vxNext_features); 

        for(size_t i = 0; i < nNextPoints; i++)
        {
            vx_keypoint_t kp1 = vxNext_features[i]; 
            vx_keypoint_t kp2 = vxPrev_features[i]; 
            openVXKeypoints.push_back(cv::KeyPoint((float)kp1.x, (float)kp1.y, 7.f, -1, kp1.strength));  
		} 
 

		std::cout << "Elapsed time over " << numberOfIterations << " OpenVX call(s): " << timer.GetElapsedUs() << " us or " << (float)timer.GetElapsedUs() / (float)numberOfIterations << "us per frame" << std::endl;

		ranOpenvx = true;
#endif

		/*------------- end of measurements ----------------------*/
		
		// compare results
		int numberOfDifferences = 0;
		double errorPerPixel = 0; 

		if (ranCuda)
		{
			std::cout << "comparing CUDA versus CPU" << std::endl;
			//imageCompare(dstCudaOnHost, tmpMatSW, numberOfDifferences, errorPerPixel, true, false);
			std::cout << "number of differences: " << numberOfDifferences << " average error per pixel: " << errorPerPixel << std::endl;
		}
		if (ranOpenvx)
		{
			std::cout << "comparing Openvx versus CPU" << std::endl;
			//imageCompare(dstOpenvx, tmpMatSW, numberOfDifferences, errorPerPixel, true, false);
			std::cout << "number of differences: " << numberOfDifferences << " average error per pixel: " << errorPerPixel << std::endl;
		}
		
		//show results
		if (imShowOn) {
       		imshow("prevImg", prevImg);              		
			imshow("currImg", currImg);       
 
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
 

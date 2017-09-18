#include <iostream>
#include <stdlib.h>
//Kinect Header
#include <Kinect.h>
#include <d2d1.h>
//OpenCV Header
#include "highgui/highgui.hpp"    
#include "opencv2/nonfree/nonfree.hpp"    
#include "opencv2/legacy/legacy.hpp"   


using namespace std;
using namespace cv;


int main()
{
	// 1a. Get default Sensor and mapping method ready.
	IKinectSensor* pSensor = nullptr;
	GetDefaultKinectSensor(&pSensor);
	ICoordinateMapper*      m_pCoordinateMapper;
	pSensor->get_CoordinateMapper(&m_pCoordinateMapper);
	// 1b. Open sensor
	pSensor->Open();
	CameraIntrinsics cameraIntrinsics;
	m_pCoordinateMapper->GetDepthCameraIntrinsics(&cameraIntrinsics);
	while (cameraIntrinsics.FocalLengthX == 0)
	{
		m_pCoordinateMapper->GetDepthCameraIntrinsics(&cameraIntrinsics);
	}
	cout << cameraIntrinsics.FocalLengthX << " " << cameraIntrinsics.FocalLengthY << endl;
	cout << cameraIntrinsics.PrincipalPointX << " " << cameraIntrinsics.PrincipalPointY << endl;
	// 2a. Get frame source 

	IColorFrameSource* rgbFrameSource = nullptr;
	//The following for getting the depth image.
	IDepthFrameSource* dFrameSource = nullptr;
	pSensor->get_ColorFrameSource(&rgbFrameSource);
	pSensor->get_DepthFrameSource(&dFrameSource);

	// 2b. Get frame description
	int        iWidth = 0;
	int        iHeight = 0;
	IFrameDescription* rgbFrameDescription = nullptr;
	rgbFrameSource->get_FrameDescription(&rgbFrameDescription);
	rgbFrameDescription->get_Width(&iWidth);
	rgbFrameDescription->get_Height(&iHeight);
	rgbFrameDescription->Release();
	rgbFrameDescription = nullptr;
	cout << "iWidth= " << iWidth << endl << "iHeight= " << iHeight << endl;

	int        dWidth = 0;
	int        dHeight = 0;
	IFrameDescription* dFrameDescription = nullptr;
	dFrameSource->get_FrameDescription(&dFrameDescription);
	dFrameDescription->get_Width(&dWidth);
	dFrameDescription->get_Height(&dHeight);
	dFrameDescription->Release();
	dFrameDescription = nullptr;
	cout << "dWidth= " << dWidth << endl << "dHeight= " << dHeight << endl;

	// 2c. get some dpeth only meta
	UINT16 uDepthMin = 0, uDepthMax = 0;
	dFrameSource->get_DepthMinReliableDistance(&uDepthMin);
	dFrameSource->get_DepthMaxReliableDistance(&uDepthMax);
	cout << "Reliable Distance: " << uDepthMin << " & " << uDepthMax << endl;

	//prepare OpenCV
	cv::Mat rgb_im, depth_im, rgb_display, depth_display;
	cv::Mat rgbd_im(iHeight, iWidth, CV_64F);

	// 3a. get frame reader
	IColorFrameReader* rgbFrameReader = nullptr;
	rgbFrameSource->OpenReader(&rgbFrameReader);

	IDepthFrameReader* dFrameReader = nullptr;
	dFrameSource->OpenReader(&dFrameReader);

	RGBQUAD* m_pColorRGBX = new RGBQUAD[iWidth * iHeight];
	RGBQUAD* m_pDepthRGBX = new RGBQUAD[dWidth * dHeight];
	
	DepthSpacePoint* m_pDepthCoordinates = new DepthSpacePoint[iWidth * iHeight];

	//Enter main loop

	while (true)
	{
		// 4a. Get last frame of RGB 
		IColorFrame* rgbFrame = nullptr;
		IDepthFrame* dFrame = nullptr;

		UINT nBufferSize = 0;
		ColorImageFormat imageFormat = ColorImageFormat_None;
		RGBQUAD *nBuffer = NULL;

		UINT    pBufferSize = 0;
		UINT16*    pBuffer = nullptr;
		RGBQUAD* pRGBX = m_pDepthRGBX;
		bool flag = false;
		if (rgbFrameReader->AcquireLatestFrame(&rgbFrame) == S_OK)
		{
			rgbFrame->get_RawColorImageFormat(&imageFormat);

			if (imageFormat == ColorImageFormat_Bgra)
			{
				rgbFrame->AccessRawUnderlyingBuffer(&nBufferSize, reinterpret_cast<BYTE**>(&nBuffer));
			}
			else if (m_pColorRGBX)
			{
				nBuffer = m_pColorRGBX;
				nBufferSize = iWidth * iHeight * sizeof(RGBQUAD);
				rgbFrame->CopyConvertedFrameDataToArray(nBufferSize, reinterpret_cast<BYTE*>(nBuffer), ColorImageFormat_Bgra);
			}
			rgb_im.create(iHeight, iWidth, CV_8UC4);
			memcpy(rgb_im.data, nBuffer, 4 * iHeight*iWidth * sizeof(BYTE));
			imwrite("ColorImage1.jpg", rgb_im);
			// release frame
			//rgbFrame->Release();
			flag = true;
		}

		// 4b. Get last frame of Depth.

		if (dFrameReader->AcquireLatestFrame(&dFrame) == S_OK)
		{
			dFrame->AccessUnderlyingBuffer(&pBufferSize, &pBuffer);
			depth_im.create(dHeight, dWidth, CV_16U);
			memcpy(depth_im.data, pBuffer, dHeight*dWidth * sizeof(UINT16));
		}

		if (flag == true) {
			break;
		}
	}

	if (m_pCoordinateMapper && m_pDepthCoordinates)
	{
		m_pCoordinateMapper->MapColorFrameToDepthSpace(dWidth * dHeight, (UINT16*)depth_im.data, iWidth * iHeight, m_pDepthCoordinates);

		for (int i = 0; i < rgbd_im.rows; i++)
		{
			for (int j = 0; j < rgbd_im.cols; j++)
			{
				rgbd_im.at<double>(i, j) = -1;
				double a = i * dWidth + j;
				DepthSpacePoint depthPoint = m_pDepthCoordinates[i * rgbd_im.cols + j];
				int depthX = (int)(floor(depthPoint.X + 0.5));
				int depthY = (int)(floor(depthPoint.Y + 0.5));
				//std::cout << "depth:(" << depthX << "," << depthY << ")" << std::endl;
				if ((depthX >= 0) && (depthX < dWidth) && (depthY >= 0) && (depthY < dHeight))
				{
				/*	if (depth_im.at<UINT16>(depthY, depthX) == 0)
						rgbd_im.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
					else*/
					{
						int depth = depth_im.at<UINT16>(depthY, depthX);
						//cout << i << " " << j << " depth= " << depth << endl;
						rgbd_im.at<double>(i, j) = depth;
					}
				}
			}
		}
	}
	//imwrite("DepthImage1", rgbd_im);
	FileStorage fs("DepthData1.xml", FileStorage::WRITE);
	fs << "depth" << rgbd_im;
	fs.release();
	cv::cvtColor(rgb_im, rgb_display, CV_BGRA2BGR);
	cv::normalize(depth_im, depth_display, 0, 255, CV_MINMAX, CV_8UC1);
	cv::Mat rgbdepth = depth_display.clone();
	cv::cvtColor(rgbdepth, rgbdepth, CV_GRAY2BGR);
	for (int iii = 0; iii < depth_display.rows; iii++)
	{
		for (int jjj = 0; jjj < depth_display.cols; jjj++)
		{
			if (depth_display.at<uchar>(iii, jjj) == 0)
			{
				rgbdepth.at<cv::Vec3b>(iii, jjj) = cv::Vec3b(0, 0, 255);
			}
		}
	}
	cv::resize(rgb_display, rgb_display, cv::Size(512, 424));
	cv::imshow("depth", rgbdepth);
	cv::imshow("rgb (scaled)", rgb_display);
	waitKey(0);
}
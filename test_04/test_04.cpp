// test_04.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace cv;

Mat calc_cdf(const Mat &input)
{
	Mat res(input);
	
	for (int i = 1; i < 256; i++)
	{
		res.at<float>(i) = res.at<float>(i - 1) + input.at<float>(i);
		
	}
	std::cout << res << std::endl;
	return res;
}

Mat calc_inv(const Mat &input)
{

}

int main(int argc, char** argv)
{
	char* src_imageName = argv[1];
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };

	Mat src_image;
	src_image = imread(src_imageName, 1);

	if (argc != 2 || !src_image.data)
	{
		printf(" No image data \n ");
		return -1;
	}

	Mat gray_image;
	cvtColor(src_image, gray_image, CV_BGR2GRAY);

	imwrite("../../Images/gray_image.jpg", gray_image);

	// calculate the gray level histogram.
	Mat src_hist;
	bool uniform = true; bool accumulate = false;
	calcHist(&gray_image, 1, 0, Mat(), src_hist, 1, &histSize, &histRange, uniform, accumulate);
	
	int hist_w = 512; int hist_h = 400; // h is rows, w is column.
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	// normalize(src_hist, src_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	std::cout << src_hist << std::endl;
	
	// compute cdf
	Mat src_cdf = calc_cdf(src_hist);
	// std::cout << gray_cdf << std::endl;

	// draw gray level hist
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(src_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(src_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}
	namedWindow("Gray Histogram", CV_WINDOW_AUTOSIZE);
	namedWindow(src_imageName, CV_WINDOW_AUTOSIZE);
	namedWindow("Gray image", CV_WINDOW_AUTOSIZE);
	
	imshow("Gray Histogram", histImage);
	imshow(src_imageName, src_image);
	imshow("Gray image", gray_image);

	waitKey(0);

	return 0;
}




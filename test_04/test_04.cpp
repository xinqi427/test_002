// test_04.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace cv;

Mat his2cdf(const Mat &input)
{
	Mat res(input);
	
	for (int i = 1; i < input.rows; i++)
	{
		res.at<float>(i) = res.at<float>(i - 1) + input.at<float>(i);
		
	}
	// std::cout << res << std::endl;
	return res;
}

//Mat calc_inv(const Mat &input)
//{
//	Mat res(input.at<float>(input.size()), 1,  CV_32F);
//
//
//}

int main(int argc, char** argv)
{
	char* src_imageName = argv[1];
	char* tgt_imageName = argv[2];
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };

	Mat src_image, tgt_image;
	src_image = imread(src_imageName, 1);
	tgt_image = imread(tgt_imageName, 1);

	if (argc != 3 || !src_image.data)
	{
		printf(" No image data \n ");
		return -1;
	}

	Mat src_gray_image, tgt_gray_image;
	cvtColor(src_image, src_gray_image, CV_BGR2GRAY);
	cvtColor(tgt_image, tgt_gray_image, CV_BGR2GRAY);

	// imwrite("../../Images/gray_image.jpg", src_gray_image);

	// calculate the gray level histogram.
	Mat src_hist, tgt_hist;
	bool uniform = true; bool accumulate = false;
	calcHist(&src_gray_image, 1, 0, Mat(), src_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&tgt_gray_image, 1, 0, Mat(), tgt_hist, 1, &histSize, &histRange, uniform, accumulate);


	int hist_w = 512; int hist_h = 400; // h is rows, w is column.
	int bin_w = cvRound((double)hist_w / histSize);

	Mat src_histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	// normalize(src_hist, src_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	// std::cout << src_hist.size() << std::endl;
	
	// compute cdf
	Mat src_cdf = his2cdf(src_hist);
	Mat tgt_cdf = his2cdf(tgt_hist);

	// draw gray level hist
	for (int i = 1; i < histSize; i++)
	{
		line(src_histImage, Point(bin_w*(i - 1), hist_h - cvRound(src_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(src_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}
	namedWindow("Gray Histogram", CV_WINDOW_AUTOSIZE);
	namedWindow(src_imageName, CV_WINDOW_AUTOSIZE);
	namedWindow("Gray image", CV_WINDOW_AUTOSIZE);
	
	imshow("Gray Histogram", src_histImage);
	imshow(src_imageName, src_image);
	imshow("Gray image", src_gray_image);

	waitKey(0);

	return 0;
}




// test_06.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

Mat his2cdf(const Mat &input)
{
	Mat res = input.clone();

	for (int i = 1; i < input.rows; i++)
	{
		res.at<float>(i) = res.at<float>(i - 1) + input.at<float>(i);
	}

	return res;
}

Mat cal_map(InputArray _src, InputArray _tgt)
{
	int i = 0, j = 0;
	Mat src = _src.getMat();
	Mat tgt = _tgt.getMat();
	Mat dst = Mat(src.size(), src.type());
	dst.setTo(0);

	for (i = 0; i < src.rows; i++)
	{
		for (; j < tgt.rows; j++)
		{
			float f1 = src.at<float>(i), t1 = tgt.at<float>(j);
			if (f1 < t1) {
				dst.at<float>(i) = j;
				break;
			}
		}
	}
	for (i = 1; i < dst.rows; i++)
	{
		if (dst.at<float>(i) < 1)
		{
			dst.at<float>(i) = dst.at<float>(i - 1);
		}
	}

	return dst;
}

Mat redesign(Mat _src, Mat _map)
{
	int i, j;
	Mat _dst = Mat(_src.size(), _src.type());
	for (i = 0; i < _src.rows; i++)
	{
		for (j = 0; j < _src.cols; j++)
		{
			int k = _map.at<float>(_src.at<uchar>(i, j));
			_dst.at<uchar>(i, j) = k;
		}
	}

	return _dst;
}

Mat compute_map(Mat src_single, Mat src_hist, Mat tgt_hist)
{
	Mat src_cdf = his2cdf(src_hist);
	Mat tgt_cdf = his2cdf(tgt_hist);
	normalize(src_cdf, src_cdf, 0, 1, NORM_MINMAX, -1, Mat());
	normalize(tgt_cdf, tgt_cdf, 0, 1, NORM_MINMAX, -1, Mat());

	Mat dst = cal_map(src_cdf, tgt_cdf);
	Mat new_single = redesign(src_single, dst);
	return new_single;
}

int main(int argc, char ** argv)
{
	char * src_Name = argv[1], * tgt_Name = argv[2];

	int histSize = 256;
	float range[] = { 0,256 };
	const float* histRange = { range };

	Mat src_image, tgt_image, dst_image;
	src_image = imread(src_Name, 1);
	tgt_image = imread(tgt_Name, 1);
	if (argc != 3 || !src_image.data || !tgt_image.data)
	{
		printf("No Image Data \n");
		return -1;
	}
	dst_image.create(src_image.rows, src_image.cols, CV_8UC1);

	

	vector<Mat> src_bgr, tgt_bgr, dst_bgr;	// split into 3 channels
	split(src_image, src_bgr);
	split(tgt_image, tgt_bgr);

	bool uniform = true, accumulate = false;

	// Mat src_b, src_g, src_r, tgt_b, tgt_g, tgt_r;	
	// calculate the hist for each channel
	vector<Mat> src_hist, tgt_hist;
	
	for (int i = 0; i < src_bgr.size(); i++)
	{
		Mat src_ele, tgt_ele;
		calcHist(&src_bgr[0], 1, 0, Mat(), src_ele, 1, &histSize, &histRange, uniform, accumulate);
		src_hist.push_back(src_ele);
		calcHist(&tgt_bgr[0], 1, 0, Mat(), tgt_ele, 1, &histSize, &histRange, uniform, accumulate);
		tgt_hist.push_back(tgt_ele);
	}
	
	for (int i = 0; i < src_bgr.size(); i++)
	{
		dst_bgr.push_back(compute_map(src_bgr[i], src_hist[i], tgt_hist[i]));
	}

	merge(dst_bgr, dst_image);

	imshow(src_Name, src_image);
	imshow(tgt_Name, tgt_image);
	imshow("New Image", dst_image);

	waitKey(0);
	;
	return 0;
}


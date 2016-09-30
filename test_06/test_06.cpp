// test_06.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

typedef Mat_<float> Array;
typedef Mat_<uchar> ArrayU;

Array his2cdf(const Array &input)
{
	Array res = input.clone();

	for (int i = 1; i < input.rows; i++)
	{
		res(i) = res(i - 1) + input(i);
	}

	return res;
}

Array cal_map(const Array &src, const Array &tgt)
{
	int i = 0, j = 0;

	Array dst = Array(src.size());
	dst.setTo(0);

	for (i = 0; i < src.rows; i++)
	{
		for (; j < tgt.rows; j++)
		{
			float f1 = src(i), t1 = tgt(j);
			if (f1 < t1) {
				dst(i) = j;
				break;
			}
		}
	}
	for (i = 1; i < dst.rows; i++)
	{
		if (dst(i) < 1)
		{
			dst(i) = dst(i - 1);
		}
	}

	return dst;
}

ArrayU redesign(Array _src, Array _map)
{
	int i, j;
	ArrayU _dst = ArrayU(_src.size());
	for (i = 0; i < _src.rows; i++)
	{
		for (j = 0; j < _src.cols; j++)
		{
			int k = _map(_src(i, j));
			_dst(i, j) = k;
		}
	}

	return _dst;
}

Array compute_map(Array src_single, Array src_hist, Array tgt_hist)
{
	Array src_cdf = his2cdf(src_hist);
	Array tgt_cdf = his2cdf(tgt_hist);
	normalize(src_cdf, src_cdf, 0, 1, NORM_MINMAX, -1, Array());
	normalize(tgt_cdf, tgt_cdf, 0, 1, NORM_MINMAX, -1, Array());

	Array dst = cal_map(src_cdf, tgt_cdf);
	Array new_single = redesign(src_single, dst);
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
	dst_image.create(src_image.rows, src_image.cols, CV_8UC3);	

	vector<ArrayU> src_bgr, tgt_bgr, dst_bgr;	// split into 3 channels
	split(src_image, src_bgr);
	split(tgt_image, tgt_bgr);

	bool uniform = true, accumulate = false;

	// Array src_b, src_g, src_r, tgt_b, tgt_g, tgt_r;	
	// calculate the hist for each channel
	vector<Array> src_hist, tgt_hist;
	
	for (int i = 0; i < src_bgr.size(); i++)
	{
		Array src_ele, tgt_ele;
		calcHist(&src_bgr[i], 1, 0, Array(), src_ele, 1, &histSize, &histRange, uniform, accumulate);
		src_hist.push_back(src_ele);
		calcHist(&tgt_bgr[i], 1, 0, Array(), tgt_ele, 1, &histSize, &histRange, uniform, accumulate);
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
	return 0;
}


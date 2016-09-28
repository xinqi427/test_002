// test_05.cpp : Defines the entry point for the console application.
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
	for ( i = 0; i < _src.rows; i++)
	{
		for ( j = 0; j < _src.cols; j++)
		{
			int k = _map.at<float>(_src.at<uchar>(i, j));
			_dst.at<uchar>(i, j) = k;
		}
	}

	return _dst;
}

Mat cal_compose(InputArray _src, InputArray _tgt)
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
				dst.at<float>(j) = f1;
				break;
			}			
		}
	}

	for (i = 1; i < dst.rows; i++)
	{
		if (dst.at<float>(i) < 0.00001)
		{
			dst.at<float>(i) = dst.at<float>(i - 1);
		}
	}
	return dst;
};

Mat cdf2hist(const Mat &cdf)
{
	Mat hist = cdf.clone();
	for (int i = 1; i < cdf.rows; i++)
	{
		hist.at<float>(i) = cdf.at<float>(i) - cdf.at<float>(i - 1);
	}
	return hist;
}

int main(int argc, char ** argv)
{
	char * src_Name = argv[1], * tgt_Name = argv[2];
	
	int histSize = 256;
	float range[] = { 0,256 };
	const float* histRange = { range };

	Mat src_image, tgt_image;
	src_image = imread(src_Name, 1);
	tgt_image = imread(tgt_Name, 1);

	if (argc != 3 || !src_image.data)
	{
		printf(" No Image Data \n");
		return -1;
	}

	// set gray level pics.
	Mat src_gray, tgt_gray;
	cvtColor(src_image, src_gray, CV_BGR2GRAY);
	cvtColor(tgt_image, tgt_gray, CV_BGR2GRAY);

	// Calculate the gray level hist and CDF.
	Mat src_hist, tgt_hist;
	bool uniform = true, accumulate = false;
	calcHist(&src_gray, 1, 0, Mat(), src_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&tgt_gray, 1, 0, Mat(), tgt_hist, 1, &histSize, &histRange, uniform, accumulate);
	

	int hist_w = 512; int hist_h = 400; // h is rows, w is column.
	int bin_w = cvRound((double)hist_w / histSize);
	Mat src_histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	Mat src_cdf = his2cdf(src_hist);
	Mat tgt_cdf = his2cdf(tgt_hist);
	Mat s_cdf_1, t_cdf_1;
	normalize(src_cdf, s_cdf_1, 0, 1, NORM_MINMAX, -1, Mat());
	normalize(tgt_cdf, t_cdf_1, 0, 1, NORM_MINMAX, -1, Mat());
	
	// Mat dst_cdf;
	// dst_cdf.create(src_cdf.size(), src_cdf.type());

	// Calculate g^(-1)(f(x))
	// dst_cdf = Mat(cal_compose(s_cdf_1, t_cdf_1));
	// dst_cdf *= src_cdf.at<float>(255);
	// Mat new_hist = cdf2hist(dst_cdf);
	// cout << new_hist << endl;
	
	Mat dst = cal_map(s_cdf_1, t_cdf_1);

	// cout << src_gray(15, 15) << endl;
	Mat new_image = redesign(src_gray, dst);
	imshow(src_Name, src_gray);
	imshow(tgt_Name, tgt_gray);
	imshow("new", new_image);

	//namedWindow("Gray Src", CV_WINDOW_AUTOSIZE);
	//imshow(src_Name, src_gray);
	//imshow("Gray Src", src_histImage);
	//imshow(tgt_Name, tgt_gray);
	//imshow("new", new_image);

	waitKey(0);

	return 0;

}


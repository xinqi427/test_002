// test_07.cpp : rotate the BGR of the image
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace std;
using namespace cv;

int main(int argc, char ** argv)
{
	char * src_Name = argv[1];
	float range[] = { 0,256 };
	const float* histRange = { range };

	Mat src_image = imread(src_Name, 1);

	/*Vec3f pixel = src_image.at<Vec3f>(13, 15);
	int b = pixel[0];
	cout << b << endl;*/

	Vec3b colour = src_image.at<Vec3b>(Point(13, 15));
	// cout << src_image.at<Vec3b>(Point(13, 15)) << endl;
	// cout << src_image.at<Vec3b>(Point(13, 15))[1] << endl;

	// compute a rotation matrix.
	Mat1d mat(3,3, CV_64FC1); // Or: Mat mat(2, 4, CV_64FC1);
	double low = -1.0;
	double high = +1.0;
	randu(mat, Scalar(low), Scalar(high));
	// cout << mat << endl;
	// cout << mat.row(0) << endl;
	Mat mat_n = Mat(3, 3, CV_64FC1);
	for (int i = 0; i < mat.rows; i++)
	{
		normalize(mat.row(i), mat_n.row(i), 1, 0, NORM_L2, -1, noArray());
	}
	
	Mat res = mat_n * Mat(colour);
	cout << res << endl;

	// cout << mat_n << endl;

	imshow(src_Name, src_image);
	waitKey(0);
	;
    return 0;
}


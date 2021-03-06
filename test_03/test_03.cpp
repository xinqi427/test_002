// test_03.cpp : Defines the entry point for the console application.
//


#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

Mat image, image_eq;
int choice = 0;

void on_trackbar(int, void*) {
	if (choice == 0) // normal image
		imshow("Image", image);
	else // histogram equalized image
		imshow("Image", image_eq);
}

int main(int argc, char** argv) {
	image = imread(argv[1], 1);
	image_eq.create(image.rows, image.cols, CV_8UC1);

	//separate channels, equalize histograms and them merge them
	vector<Mat> channels, channels_eq;

	split(image, channels);

	for (int i = 0; i < channels.size(); i++) {
		Mat eq;
		equalizeHist(channels[i], eq);
		channels_eq.push_back(eq);
	}

	merge(channels_eq, image_eq);

	namedWindow("Image");

	createTrackbar("Normal/Eq.", "Image", &choice, 1, on_trackbar);
	on_trackbar(0, 0);

	while (char(waitKey(1)) != 'q') {}

	return 0;
}

// IterativeDistributionTransfer.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "IDT.h"
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

int main(int argc, char ** argv)
{
	char * src_name = argv[1], *tgt_name = argv[2];

	Image src, tgt, dst;
	src = imread(src_name);
	tgt = imread(tgt_name);

	IDT idt(src, tgt);
	dst = idt.compute();

	imshow("New Image", dst);
	waitKey(0);

	return 0;
}


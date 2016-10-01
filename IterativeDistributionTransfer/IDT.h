#pragma once

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

typedef Mat_<Vec<uchar, 3>> Image;
typedef Mat_<Vec<float, 3>> Imagef;
typedef Mat_<float> Array;
typedef Mat_<int> ArrayI;
typedef Mat_<uchar> ArrayU;
typedef Mat_<float> Rotation;

class IDT
{
public:
	IDT(const Image& src, const Image& tgt);
	Image compute();
	~IDT();

private:
	typedef Mat_<Vec<float, 3>> Imagef;
	Array omt1d(const Array& src, const Array& tgt);
	float compute_map();
	void compute_hist();
	Imagef _tmp;
	Image _dst;
	const Image& _src, _tgt;
	vector<Array> _src_hist, _tgt_hist, _dst_hist;	
	vector<Array> _src_bgr, _tgt_bgr, _dst_bgr;
	Rotation R;
};

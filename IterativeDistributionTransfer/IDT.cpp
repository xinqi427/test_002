#include "stdafx.h"
#include "IDT.h"
#include <opencv2/imgproc/imgproc.hpp>

Rotation generate_random_rotation(int n)
{
	Rotation R(n, n);
	double low = 0;
	double high = 1.0;
	randu(R, Scalar(low), Scalar(high));

	for (auto i = 0; i < R.cols; i++)
	{
		Array r = R.col(i);
		for (int j = 0; j < i; j++)
		{
			Array b = R.col(j);
			r -= b*(b.t()*r);
		}
		normalize(r, r);
	}

	return R;
}


IDT::IDT(const Image &src, const Image &tgt) :_src(src), _tgt(tgt)
{
	Imagef _src2(src.size()), _tgt2(tgt.size());
	_src.convertTo(_src2, _src2.type());
	_tgt.convertTo(_tgt2, _tgt2.type());
	normalize(_src2, _src2, 0, 4095, NORM_MINMAX, -1);
	normalize(_tgt2, _tgt2, 0, 4095, NORM_MINMAX, -1);
	split(_src2, _src_bgr);
	split(_tgt2, _tgt_bgr);
}

Image IDT::compute() {

	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);

	float eps = 1e-6;
	float err = 1;
	int k = 0;

	while (err > eps && k++ < 20)
	{
		err = compute_map();
		string out_name("tmp\\a");
		out_name += to_string(k) + ".jpg";

		try 
		{
			imwrite(out_name, _dst);
		}
		catch (cv::Exception& ex) {
			fprintf(stderr, "%s\n", ex.what());
			return _dst;
		}

	}

	return _dst;
}


IDT::~IDT()
{
}


Array hist2cdf(const ArrayI &input, bool normalized = true)
{
	Array res = Array::zeros(input.size());

	for (auto i = 1; i < input.rows; i++)
	{
		res(i) = res(i - 1) + input(i);
	}

	if (normalized) {
		normalize(res, res, 0, 1, NORM_MINMAX, -1);
	}

	return res;
}

Array IDT::omt1d(const Array &src, const Array &tgt)
{
	Array dst = Array::zeros(src.size());

	/*for (int i = 0, j = 0; i < src.rows; ++i)
	{
		float fi = src(i);
		for (j = 0; j < tgt.rows; ++j)
		{
			float fj = tgt(j);
			
			if (fi < fj)
			{
				dst(i) = j;
				break;
			}
			
		}
		
	}*/

	for (auto i = 0, j = 0; i < src.rows; i++)
	{
		for (; j < tgt.rows; j++)
		{
			float f1 = src(i), t1 = tgt(j);
			if (f1 < t1)
			{
				dst(i) = j;
				break;
			}
		}
	}

	for (auto i = 1; i < dst.rows; i++)
	{
		if (dst(i) < 1)
		{
			dst(i) = dst(i - 1);
		}
	}

	return dst;
}

ArrayU retarget(Array _src, Array _map)
{
	ArrayU _dst = ArrayU(_src.size());
	for (auto i = 0; i < _src.rows; i++)
	{
		for (auto j = 0; j < _src.cols; j++)
		{
			_dst(i, j) = _map(_src(i, j));
		}
	}

	return _dst;
}

void IDT::compute_hist()
{
	int histSize[] = { 4096 };
	float range[] = { 0, 4096 };
	const float* histRange[] = { range };
	_src_hist.clear();
	_tgt_hist.clear();
	for (int i = 0; i < _src_bgr.size(); i++)
	{
		Array sh, th;
		calcHist(&_src_bgr[i], 1, 0, Array(), sh, 1, histSize, histRange);
		_src_hist.push_back(sh);
		calcHist(&_tgt_bgr[i], 1, 0, Array(), th, 1, histSize, histRange);
		_tgt_hist.push_back(th);
	}
}

void rotate_array(vector<ArrayU>& array, const Rotation& R)
{
	vector<Array> out;
	for (auto i = 0; i < array.size(); ++i)
	{
		Array ai = Array::zeros(array[i].size());
		for (auto j = 0; j < array.size(); ++j)
		{
			Array tj;
			array[j].convertTo(tj, tj.type());
			multiply(tj, Scalar::all(R(j, i)), tj, ai.type());
			ai += tj;
		}
		out.push_back(ai);
	}
	for (auto i = 0; i < array.size(); ++i)
	{
		normalize(out[i], out[i], 0, 255, NORM_MINMAX);
		out[i].copyTo(array[i], ArrayU());

		//cout << array[i] << endl;
	}

}

void rotate_array(vector<Array>& array, const Rotation& R)
{
	vector<Array> out;
	for (auto i = 0; i < array.size(); ++i)
	{
		Array ai = Array::zeros(array[i].size());
		for (auto j = 0; j < array.size(); ++j)
		{
			Array tj;
			array[j].convertTo(tj, tj.type());
			multiply(tj, Scalar::all(R(j, i)), tj, ai.type());
			ai += tj;
		}
		out.push_back(ai);
	}
	for (auto i = 0; i < array.size(); ++i)
	{
		normalize(out[i], out[i], 0, 4095, NORM_MINMAX);
		out[i].copyTo(array[i]);
	}

}



float IDT::compute_map()
{
	R = generate_random_rotation(_src_bgr.size());

	R = Rotation::eye(R.size());
	Rotation Rt;
	transpose(R, Rt);

	rotate_array(_src_bgr, Rt);
	rotate_array(_tgt_bgr, Rt);

	compute_hist();
	_dst_bgr.clear();

	for (auto i = 0; i < _src_bgr.size(); i++)
	{
		Array src_cdf = hist2cdf(_src_hist[i]);
		Array tgt_cdf = hist2cdf(_tgt_hist[i]);
		normalize(src_cdf, src_cdf, 0, 1, NORM_MINMAX);
		normalize(tgt_cdf, tgt_cdf, 0, 1, NORM_MINMAX);
		Array map = omt1d(src_cdf, tgt_cdf);
		Array map2;
		transpose(map, map2);
		//cout << _src_hist[i] << endl;
		//cout << _tgt_hist[i] << endl;
		cout << map2 << endl;
		_dst_bgr.push_back(retarget(_src_bgr[i], map));
	}
	rotate_array(_dst_bgr, R);
	_tmp.setTo(Scalar(0));
	merge(_dst_bgr, _tmp);
	normalize(_tmp, _tmp, 0, 255, NORM_MINMAX, -1);
	_tmp.convertTo(_dst, _dst.type());
	

	for (auto i = 0; i < _src_bgr.size(); i++)
	{
		_dst_bgr[i].copyTo(_src_bgr[i]);
	}

	return 1;

}

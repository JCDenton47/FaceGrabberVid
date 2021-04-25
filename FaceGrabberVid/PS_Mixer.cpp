#include "PS_Mixer.h"

using namespace std;
using namespace cv;

bool CheckInput(const cv::Mat& src, const cv::Mat& mask)
{
	//检查源图像的通道数
	if (src.channels() < 3)
	{
		cout << "error in color_mixer, you must input a 3 channel pic" << endl;
		return false;
	}
	else if (src.empty() || mask.empty())
	{
		cout << "input or mask empty" << endl;
		return false;
	}
	return true;
}

//bool ColorMixer::Mix(const cv::Mat& src, const cv::Mat& mask, cv::Mat& output)
//{
//	//检查输入
//	if (!CheckInput(src, mask))
//		return false;
//	//转换源图像和mask为HSV
//	Mat src_hsv, mask_hsv;
//	cvtColor(src, src_hsv, COLOR_BGR2HSV);
//	cvtColor(mask, mask_hsv, COLOR_BGR2HSV);
//	//给output图像内存
//	output.create(Size(src.size()), src.type());
//
//	//遍历图像进行赋值
//	for (int y = 0; y < src.rows; ++y)
//	{
//		for (int x = 0; x < src.cols; ++x)
//		{
//			//HcScBc = HaSaBb
//			const auto& pixel_a = mask_hsv.at<Vec3b>(y, x);
//			const auto& pixel_b = src_hsv.at<Vec3b>(y, x);
//			output.at<Vec3b>(y, x) = { pixel_a[0], pixel_a[1], pixel_b[2] };
//		}
//	}
//	cvtColor(output, output, COLOR_HSV2BGR);
//
//	return true;
//}

//bool ColorMixer::Release()
//{
//	delete this;
//	return true;
//}
//
//MixerPtr MixerFactory::GetMixer(const std::string& mix_mode)
//{
//	if (mix_mode == MIX_TYPE::COLOR)
//	{
//		return MixerPtr(new ColorMixer, mem_fun(&ColorMixer::Release));
//	}
//}

void PhotoShop::FillContour(cv::Mat& input, cv::Mat& output, const uchar mask_val)
{
	if (input.empty())
	{
		return;
	}
	Mat img;

	//对轮廓图进行填充
	if (input.type() != CV_8UC1)
	{
		cvtColor(input, img, COLOR_BGR2GRAY);
	}
	else
	{
		img = input.clone();
	}
	output = Mat::zeros(img.size(), img.type());


	std::vector<std::vector<Point>> contours;
	findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	//对图中的所有轮廓进行填充
	for (const auto& contour : contours)
	{
		Mat tmp = Mat::zeros(img.size(), img.type());

		Rect b_rect = boundingRect(contour);
		b_rect.x = max(0, b_rect.x - 1);
		b_rect.y = max(0, b_rect.y - 1);
		b_rect.width += 2;
		if (b_rect.width + b_rect.x > img.cols)
			b_rect.width = img.cols - 1 - b_rect.x;
		if (b_rect.height + b_rect.y > img.rows)
			b_rect.height = img.rows - 1 - b_rect.y;


		auto fun_in_rect = [&b_rect](int x, int y)
		{
			return (x >= b_rect.x && x <= b_rect.x + b_rect.width && y >= b_rect.y && y <= b_rect.y + b_rect.height);
		};
		std::queue<Point> neighbor_queue;
		neighbor_queue.emplace(b_rect.x, b_rect.y);
		tmp.at<uchar>(b_rect.y, b_rect.x) = 128;

		while (!neighbor_queue.empty())
		{
			//从队列取出种子点，获取其4邻域坐标点
			auto seed = neighbor_queue.front();
			neighbor_queue.pop();

			std::vector<Point> pts;
			pts.emplace_back(seed.x, (seed.y - 1));
			pts.emplace_back(seed.x, (seed.y + 1));
			pts.emplace_back((seed.x - 1), seed.y);
			pts.emplace_back((seed.x + 1), seed.y);

			for (auto& pt : pts)
			{
				if (fun_in_rect(pt.x, pt.y) && tmp.at<uchar>(pt.y, pt.x) == 0 && img.at<uchar>(pt.y, pt.x) == 0)
				{
					//将矩形范围内且灰度值为0的可连通坐标点添加到队列
					neighbor_queue.push(pt);
					tmp.at<uchar>(pt.y, pt.x) = 128;
				}
			}

		}


		for (int i = b_rect.y; i < b_rect.y + b_rect.height; i++)
		{
			for (int j = b_rect.x; j < b_rect.x + b_rect.width; j++)
			{
				if (tmp.at<uchar>(i, j) == 0)
				{
					output.at<uchar>(i, j) = mask_val;
				}
			}
		}
	}
	return;
}


void PhotoShop::GetFilledMask(cv::Mat& input, cv::Mat& mask, bool all_drawn)
{
	Mat tmp = input.clone();
	if (input.channels() == 3)
		cvtColor(input, tmp, COLOR_BGR2GRAY);
	threshold(tmp, tmp, 50, 255, THRESH_BINARY);

	vector<vector<Point>> contours;
	int index = 0;
	findContours(tmp, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);


	if (!all_drawn)
	{
		for (int i = 1; i < contours.size(); ++i)
		{
			if (contours[i].size() > contours[index].size())
				index = i;
		}
		drawContours(mask, contours, index, Scalar(255), -1);
	}
	else
	{
		drawContours(mask, contours, -1, Scalar(255), -1);
	}


}
//美颜处理
void PhotoShop::FaceBeautify(Mat& input, Mat& output)
{


	Mat dst_grinded;
	FaceGrinding(input, dst_grinded);
	Mat dst_Saturated(input.size(), input.type());
	AdjustSaturation(dst_grinded, dst_Saturated);
	Mat dst_brighted(input.size(), input.type());
	AdjustBrightness(dst_Saturated, dst_brighted);
	//转换为三通道
	cvtColor(dst_brighted, dst_brighted, COLOR_BGRA2BGR);
	output = dst_brighted.clone();
}

bool PhotoShop::GetBaldHead(cv::Mat src, cv::Mat input_mask, cv::Mat output_mask, cv::Mat& dst)
{
	assert(!input.empty() && !input_mask.empty() && !output_mask.empty());
	assert(input.channels() == 3 && input_mask.channels() == 1 && output_mask.channels() == 1);

	Mat input = src.clone();
	//放入两个东西， 脸mask, 以及对应的彩图
	//计算平均肤色
	//将皮肤边缘作为第一批的种子点
	//处理种子点
	/*
	* 第一步，处理队列中的种子点
	* 如果种子点有值，就将旁边的无值点存起来， 并将存入点的位置在反向mask中的值置零
	* 如果种子点无值，就将该点在彩色照中的位置置为邻域均值， 并存放邻域均值到队列
	* 如果种子点旁边的有值点太少，比如少于5，存到队列末端
	* 如果有足够的种子点，算出有值点的平均像素值后赋给这个点，弹出这个点
	* 一直迭代，直到队列为空
	*/

	Mat remaining_areas;
	bitwise_not(input_mask, remaining_areas);
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	morphologyEx(remaining_areas, remaining_areas, MORPH_CLOSE, element, Point(-1, -1), 4);
	imshow("input", input);
	imshow("remain", remaining_areas);

	vector<vector<Point>> mask_edges;
	findContours(input_mask, mask_edges, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	vector<Point>& seeds = mask_edges[0];
	for (int i = 1; i < mask_edges.size(); ++i)
	{
		if (mask_edges[i].size() > seeds.size())
		{
			seeds = mask_edges[i];
		}
	}

	//预防种子太少
	if (seeds.size() < 50)
		return false;

	//存放剩下的点
	queue<Point> remain_seeds;
	for (const auto& seed : seeds)
	{
		remain_seeds.push(seed);
	}
	const int dir[][2] =
	{
		{0, -1}, {-1, -1}, {-1, 0},
		{-1, 1}, {0, 1}, {1, 1},
		{1, 0},  {1, -1} ,
	};

	auto save_region = [&remain_seeds, &remaining_areas, &dir](const Point& point)
	{
		for (int i = 0; i < 8; ++i)
		{

			Point cur_point(point.x + dir[i][0], point.y + dir[i][1]);
			//判断点是否在图中
			if (cur_point.x >= 0 && cur_point.x < remaining_areas.cols && cur_point.y >= 0 && cur_point.y < remaining_areas.rows)
			{
				//判断点的mask是否为真
				if (remaining_areas.at<uchar>(cur_point))
				{
					remain_seeds.push(cur_point);
					remaining_areas.at<uchar>(cur_point) = 0;
				}
			}
		}
	};
	auto get_region_color = [&remain_seeds, &input, &remaining_areas, &dir](const Point& point)
	{
		unsigned int blue = 0, green = 0, red = 0;
		unsigned int num = 0;
		vector<Point> vec_point;
		//for (int i = 0; i < 8; ++i)
		//{
		//	Point cur_point(point.x + dir[i][0], point.y + dir[i][1]);
		//	//判断点是否在图中
		//	if (cur_point.x >= 0 && cur_point.x < remaining_areas.cols && cur_point.y >= 0 && cur_point.y < remaining_areas.rows)
		//	{
		//		auto cur_pixel = tmp.at<Vec3b>(cur_point);
		//		//判断该点是否有颜色
		//		if (cur_pixel[0])
		//		{
		//			blue += cur_pixel[0];
		//			green += cur_pixel[1];
		//			red += cur_pixel[2];
		//			++num;
		//		}
		//	}
		//}

		for (int j = -5; j < 5; ++j)
		{
			for (int i = -5; i < 5; ++i)
			{
				Point cur_point(point.x + j, point.y + i);
				//判断点是否在图中
				if (cur_point.x >= 0 && cur_point.x < remaining_areas.cols && cur_point.y >= 0 && cur_point.y < remaining_areas.rows)
				{
					auto cur_pixel = input.at<Vec3b>(cur_point);
					//判断该点是否有颜色
					if (cur_pixel[0])
					{
						blue += cur_pixel[0];
						green += cur_pixel[1];
						red += cur_pixel[2];
						++num;
					}
				}
			}
		}


		if (num)
		{
			return Vec3b{ (unsigned char)(blue / num), (unsigned char)(green / num), (unsigned char)(red / num) };
		}
		else
		{
			/*remain_seeds.push(point);
			remaining_areas.at<uchar>(point) = 255;*/
			return Vec3b{ 0, 0, 0 };
		}
	};
	while (!remain_seeds.empty())
	{
		Point p = remain_seeds.front();
		remain_seeds.pop();

		//如果该点有值
		uchar val = input.at<Vec3b>(p)[0];
		if (!val)
		{
			//彩图有值操作无，存入附近的无值点，反向mask置零
			//无值操作
			//该点赋均值，mask赋值，存入附近的无值点，反向mask置零
			Vec3b mean_color = get_region_color(p);
			Vec3b& cur_pixel_color = input.at<Vec3b>(p);
			cur_pixel_color = mean_color;
		}

		save_region(p);

	}

	imshow("input_after", input);
	dst = Mat::zeros(input.size(), CV_8UC1);
	input.copyTo(dst, output_mask);
	imshow("dst", dst);
}

void PhotoShop::SmoothEdges(cv::Mat& img, cv::Mat& mask, cv::Mat& dst)
{
	Mat mask_dialated;
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(20, 20));
	dilate(mask, mask_dialated, element);
	Mat blur_img;
	img.copyTo(blur_img);
	blur(blur_img, blur_img, Size(5, 5));
	Mat output = img.clone();
	img.copyTo(output, mask_dialated);
	dst = output.clone();
}

//void PhotoShop::GetBaldHead(cv::Mat& input, cv::Mat& hair_only, cv::Scalar& skin_color, std::vector<cv::Rect>& eyes, std::vector<cv::Point> land_marks, cv::Mat& dst)
//{
//	try
//	{
//
//		if (input.empty() || eyes.empty())
//			return;
//
//		//确定左右眼的中心
//		Point lefteye_center, righteye_center;
//		if (eyes[0].x < eyes[1].x)
//		{
//			lefteye_center = SimpleMath::GetMidpt(eyes[0].tl(), eyes[0].br());
//			righteye_center = SimpleMath::GetMidpt(eyes[1].tl(), eyes[1].br());
//		}
//		else
//		{
//			lefteye_center = SimpleMath::GetMidpt(eyes[1].tl(), eyes[1].br());
//			righteye_center = SimpleMath::GetMidpt(eyes[0].tl(), eyes[0].br());
//		}
//		//将输入图像暂存起来
//		Mat tmp(input.clone());
//
//		//确认椭圆参数
//		//眉毛中心作为椭圆中点
//		Point2d center = SimpleMath::GetMidpt(land_marks[22], land_marks[23]);
//		Point2d lefteyebrows_center = land_marks[20], righteyebrows_center = land_marks[21];
//		//由于图像的坐标系跟笛卡尔刚好是水平相反的，所以角度需要进行加-号进行偏置
//		Point2d virtual_top = SimpleMath::GetRotatedVecRad(center, Point2d((double)land_marks[31].x, (double)land_marks[31].y), -M_PI / 2, 1.0);
//		//Point2d virtual_bottom = SimpleMath::GetRotatedVecRad(center, Point2d((double)land_marks[].x, (double)center.y), M_PI / 2, 1.2);
//		Point2d virtual_left = SimpleMath::GetRotatedVecRad(center, Point2d((double)lefteyebrows_center.x, (double)lefteyebrows_center.y), 0.01, 1.2);
//		Point2d virtual_right = SimpleMath::GetRotatedVecRad(center, Point2d((double)righteyebrows_center.x, (double)righteyebrows_center.y), 0.01, 1.2);
//
//		/*Point2d virtual_top = SimpleMath::GetRotatedVecRad(center, Point2d((double)righteye_center.x, (double)righteye_center.y), -M_PI / 2, 1.2);
//		Point2d virtual_bottom = SimpleMath::GetRotatedVecRad(center, Point2d((double)righteye_center.x, (double)righteye_center.y), M_PI / 2, 1.2);
//		Point2d virtual_left = SimpleMath::GetRotatedVecRad(center, Point2d((double)lefteye_center.x, (double)lefteye_center.y), 0.01, 1.2);
//		Point2d virtual_right = SimpleMath::GetRotatedVecRad(center, Point2d((double)righteye_center.x, (double)righteye_center.y), 0.01, 1.2);*/
//
//		//std::vector<Point2d> virtual_pts = { virtual_top, virtual_bottom, virtual_left, virtual_right };
//
//		/*circle(tmp, center, 3, Scalar(255, 0, 0), -1);
//		circle(tmp, virtual_top, 3, Scalar(0, 255, 255), -1);
//		circle(tmp, lefteye_center, 3, Scalar(0, 255, 255), -1);
//		circle(tmp, righteye_center, 3, Scalar(0, 255, 255), -1);*/
//
//		//对长短轴进行放缩，这里要改一改，至少长轴的长度需要与脸的宽度做一个比较
//		double short_axis = SimpleMath::GetLineLen(center, virtual_top) * 2.0;
//		double long_axis = SimpleMath::GetLineLen(center, virtual_left) * 2.2;
//
//		std::vector<Point> ellipes_verti;
//		//根据图像坐标系，我们其实画的下半圆
//		ellipse2Poly((Point)center, Size(long_axis, short_axis), 0, 180 + 15, 180 + 165, 1, ellipes_verti);
//
//
//		Mat mask(input.size(), CV_8UC1, Scalar(0));
//
//		for (int i = 0; i < ellipes_verti.size() - 1; ++i)
//		{
//			//line(tmp, ellipes_verti[i], ellipes_verti[i + 1], Scalar(123, 45, 78), 2);
//			cv::line(mask, ellipes_verti[i], ellipes_verti[i + 1], Scalar(255), 2);
//		}
//
//		Point2d br_padding = ellipes_verti.front(); br_padding.y = input.rows - 1;
//		Point2d bl_padding = ellipes_verti.back();  bl_padding.y = input.rows - 1;
//
//		/*line(tmp, ellipes_verti.front(), br_padding, Scalar(123, 45, 78), 2);
//		line(tmp, ellipes_verti.back(), bl_padding, Scalar(123, 45, 78), 2);
//		line(tmp, bl_padding, br_padding, Scalar(123, 45, 78), 2);*/
//
//		cv::line(mask, ellipes_verti.front(), br_padding, Scalar(255), 2);
//		cv::line(mask, ellipes_verti.back(), bl_padding, Scalar(255), 2);
//		cv::line(mask, bl_padding, br_padding, Scalar(255), 2);
//
//		Mat hair_only_tmp = hair_only.clone();
//		//染发
//		for (int x = 0; x < hair_only.cols; ++x)
//		{
//			for (int y = 0; y < hair_only.rows; ++y)
//			{
//				Vec3b& pixel_color = hair_only.at<cv::Vec3b>(y, x);
//				if (pixel_color[0] != 0 && pixel_color[1] != 0 && pixel_color[2] != 0)
//				{
//					Vec3b& new_color = hair_only_tmp.at<cv::Vec3b>(y, x);
//					new_color[0] = skin_color[0];
//					new_color[1] = skin_color[1];
//					new_color[2] = skin_color[2];
//				}
//			}
//		}
//
//		tmp += hair_only_tmp;
//
//		FillContour(mask, mask);
//
//		//将图片转换为三通道执行与运算
//		cvtColor(mask, mask, COLOR_GRAY2BGR);
//		bitwise_and(tmp, mask, dst);
//
//
//
//	}
//	catch (...)
//	{
//		//do nothing
//	}
//
//}

//滤波
void PhotoShop::FaceGrinding(Mat& input, Mat& output, int value1, int value2)
{
	int dx = value1 * 5;    //双边滤波参数之一  
	double fc = value1 * 12.5; //双边滤波参数之一  
	int transparency = 50; //透明度  
	cv::Mat dst;
	//双边滤波  
	bilateralFilter(input, dst, dx, fc, fc);
	dst = (dst - input + 128);
	//高斯模糊  
	GaussianBlur(dst, dst, cv::Size(2 - 1, 2 - 1), 0, 0);
	dst = input + 2 * dst - 255;
	dst = (input * (100 - transparency) + dst * transparency) / 100;
	dst.copyTo(output);
}
//调节对比度和亮度
void PhotoShop::AdjustSaturation(cv::Mat& input, cv::Mat& output, int saturation, const int max_increment)
{

	float increment = (saturation - 80) * 1.0 / max_increment;


	for (int col = 0; col < input.cols; col++)
	{
		for (int row = 0; row < input.rows; row++)
		{
			// R,G,B 分别对应数组中下标的 2,1,0
			uchar r = input.at<Vec3b>(row, col)[2];
			uchar g = input.at<Vec3b>(row, col)[1];
			uchar b = input.at<Vec3b>(row, col)[0];

			float maxn = max(r, max(g, b));
			float minn = min(r, min(g, b));

			float delta, value;
			delta = (maxn - minn) / 255;
			value = (maxn + minn) / 255;

			float new_r, new_g, new_b;

			if (delta == 0)		 // 差为 0 不做操作，保存原像素点
			{
				output.at<Vec3b>(row, col)[0] = b;
				output.at<Vec3b>(row, col)[1] = g;
				output.at<Vec3b>(row, col)[2] = r;
				continue;
			}

			float light, sat, alpha;
			light = value / 2;

			if (light < 0.5)
				sat = delta / value;
			else
				sat = delta / (2 - value);

			if (increment >= 0)
			{
				if ((increment + sat) >= 1)
					alpha = sat;
				else
				{
					alpha = 1 - increment;
				}
				alpha = 1 / alpha - 1;
				new_r = r + (r - light * 255) * alpha;
				new_g = g + (g - light * 255) * alpha;
				new_b = b + (b - light * 255) * alpha;
			}
			else
			{
				alpha = increment;
				new_r = light * 255 + (r - light * 255) * (1 + alpha);
				new_g = light * 255 + (g - light * 255) * (1 + alpha);
				new_b = light * 255 + (b - light * 255) * (1 + alpha);
			}
			output.at<Vec3b>(row, col)[0] = new_b;
			output.at<Vec3b>(row, col)[1] = new_g;
			output.at<Vec3b>(row, col)[2] = new_r;
		}
	}
}
//调节色调
void PhotoShop::AdjustBrightness(cv::Mat& input, cv::Mat& output, float alpha, float beta)
{
	int height = input.rows;//求出src的高
	int width = input.cols;//求出input的宽
	output = cv::Mat::zeros(input.size(), input.type());  //这句很重要，创建一个与原图一样大小的空白图片              
	//循环操作，遍历每一列，每一行的元素
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			if (input.channels() == 3)//判断是否为3通道图片
			{
				//将遍历得到的原图像素值，返回给变量b,g,r
				float b = input.at<Vec3b>(row, col)[0];//nlue
				float g = input.at<Vec3b>(row, col)[1];//green
				float r = input.at<Vec3b>(row, col)[2];//red
				//开始操作像素，对变量b,g,r做改变后再返回到新的图片。
				output.at<Vec3b>(row, col)[0] = cv::saturate_cast<uchar>(b * alpha + beta);
				output.at<Vec3b>(row, col)[1] = cv::saturate_cast<uchar>(g * alpha + beta);
				output.at<Vec3b>(row, col)[2] = cv::saturate_cast<uchar>(r * alpha + beta);
			}
			else if (input.channels() == 1)//判断是否为单通道的图片
			{

				float v = input.at<uchar>(row, col);
				output.at<uchar>(row, col) = cv::saturate_cast<uchar>(v * alpha + beta);
			}
		}
	}
}

//void PhotoShop::ApplyMask(const std::string& mask_type, const cv::Mat& input, const cv::Mat& mask, cv::Mat& dst)
//{
//	MixerFactory m_factory;
//	auto mixer = m_factory.GetMixer(mask_type);
//	Mat tmp;
//	mixer->Mix(input, mask, tmp);
//
//	double alpha = 0.7;
//	addWeighted(input, alpha, tmp, 1 - alpha, 0, dst);
//
//	cv::imshow("dst", dst);
//}




//去除背景，使背景变透明,且只保留关键位置
void PhotoShop::RemoveBackground(cv::Mat& src, cv::Mat& mask, cv::Mat& dst, bool resize_to_fit)
{
	assert(!src.empty() && !mask.empty() && !dst.empty());

	bool first_point_found = false;
	int min_x = 0, max_x = 0, min_y = 0, max_y = 0;
	Mat img = src.clone();

	if (img.channels() == 3)
	{
		cv::cvtColor(img, img, cv::COLOR_BGR2BGRA);
	}


	dst = Mat::zeros(img.size(), CV_8UC4);
	//dst = Mat::zeros(img.size(), CV_8UC3);
	img.copyTo(dst, mask);




	if (resize_to_fit)
	{
		for (int y = 0; y < img.rows; ++y)
		{
			for (int x = 0; x < img.cols; ++x)
			{
				if (mask.at<uchar>(y, x) && first_point_found == false)
				{
					min_x = x, min_y = y;
					max_x = x, max_y = y;
					first_point_found = true;
					continue;
				}
				else if (first_point_found && mask.at<uchar>(y, x))
				{
					if (x < min_x)
						min_x = x;
					if (x > max_x)
						max_x = x;
					if (y < min_y)
						min_y = y;
					if (y > max_y)
						max_y = y;
				}

			}
		}

		Rect interest_rect(min_x, min_y, max_x - min_x, max_y - min_y);
		imshow("mask", mask);
		imshow("resized", dst(interest_rect));
		dst = dst(interest_rect).clone();
	}
	cv::imshow("dst", dst);
}

XIU_Beautify* XIU_Beautify::GetInstance()
{
	static XIU_Beautify xb;

	return &xb;
}

bool XIU_Beautify::Beautify(const cv::Mat src, cv::Mat output)
{
	assert(!src.empty() && src.channels() == 3);
	Mat input;
	cv::cvtColor(src, input, cv::COLOR_BGR2BGRA);

	cv::imshow("before", input);

	Beauty_ClearParams(handle_);
	Beauty_SetWhitenRatio(handle_, 40);
	Beauty_SetSoftenRatio(handle_, 50);
	Beauty_SetFilterID(handle_, 2);
	Beauty_SetFilterRatio(handle_, 40);


	for (int i = 0; i < 2; ++i)
	{
		Beauty_ProcessBuffer(handle_, input.data, input.cols, input.rows, input.step);
	}

	cv::imshow("after", input);
	cvtColor(input, input, COLOR_BGRA2BGR);
	input.copyTo(output);
}

bool XIU_Beautify::Release()
{
	if (handle_ != NULL)
		Beauty_UninitBeautyEngine(handle_);
	return true;
}

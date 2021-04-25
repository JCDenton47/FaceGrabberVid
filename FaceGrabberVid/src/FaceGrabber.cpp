#include "FaceGrabber.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;


//打开相机
bool FaceGrabber::StarGrab()
{

	cap_.open(0, CAP_DSHOW);

	if (!cap_.isOpened())
	{
		cap_.open(1, CAP_DSHOW);
		if (!cap_.isOpened())
		{
			cout << "error cam open failed" << endl;
			return false;
		}
	}

	return true;

}

//读入一帧帧图片
void FaceGrabber::GetFrame()
{
	cap_ >> src_;
}





bool FaceGrabber::ProcessFace()
{

	if (!GetFace())
		return false;

	//脸部land_marks的位置集
	std::vector<cv::Point> land_marks;
	GetFacialLandMarks(roi_face_all_, land_marks);

	if (!GetFacialFeatures(roi_face_all_, land_marks))
	{
		return false;
	}
	if (!IsFaceStable(land_marks))
		return false;

	RotateFace(land_marks, roi_face_all_);

	land_marks.clear();

	//更新旋转后的特征点
	GetFacialLandMarks(roi_face_all_, land_marks);


	//识别性别
	if (!ProcessGender(roi_face_all_))
		return false;

	//像素语义分割
	FaceDetectTorch(roi_face_all_);//产生掩膜图dst_torch_

	//美化图片
	xiu_ptr = XIU_Beautify::GetInstance();
	xiu_ptr->Beautify(roi_face_all_, roi_face_all_);


	//获得未处理的 头脸mask, 头发mask,脸部mask, 68外边缘mask, bald_head_mask  
	GetSegments(land_marks);
	//imshow("bald", bald_head_);

	//放大mask到三倍大，腐蚀，膨胀

	float expand_ratio = 4.0;

	Mat tmp_mask = mask_face_hair_.clone();
	resize(tmp_mask, tmp_mask, Size(), expand_ratio, expand_ratio);

	Mat se_d = getStructuringElement(MORPH_ELLIPSE, Size(100, 50));
	//Mat se_e = getStructuringElement(MORPH_ELLIPSE, Size(10, 20));

	//morphologyEx(tmp_mask, tmp_mask, MORPH_ERODE, se_e, Point(-1, -1), 5);
	morphologyEx(tmp_mask, tmp_mask, MORPH_ERODE, se_d, Point(-1, -1), 5);

	resize(tmp_mask, tmp_mask, Size(), 1 / expand_ratio, 1 / expand_ratio);

	resize(roi_face_hair_, roi_face_hair_, Size(), expand_ratio, expand_ratio);
	blur(roi_face_hair_, roi_face_hair_, Size(5, 5));
	resize(roi_face_hair_, roi_face_hair_, Size(), 1 / expand_ratio, 1 / expand_ratio);



	roi_face_all_.copyTo(roi_face_hair_, tmp_mask);
	imshow("roi_face_hair", roi_face_hair_);
	//medianBlur(roi_face_hair_, roi_face_hair_, 5);



	if (!PhotoShop::GetBaldHead(roi_face_only_, mask_face_only_, mask_bald_head_, bald_head_))
	{
		Release();
		return false;
	}

	/*GetSkinColor(roi_face_only_);

	PhotoShop::FaceBeautify(roi_face_hair_, roi_face_hair_);
	PhotoShop::FaceBeautify(roi_face_all_, roi_face_all_);

	PhotoShop::FaceBeautify(bald_head_, bald_head_);*/



	return true;
}


bool FaceGrabber::GetFace()
{
	//src判空
	bool face_dectected = false;
	if (src_.empty())
		return false;

	//整体像素值减去平均值（mean）通过缩放系数（scalefactor）对图片像素值进行缩放
	cv::Mat blob_image = blobFromImage(src_, 1.0,
		cv::Size(300, 300),
		cv::Scalar(104.0, 177.0, 123.0), false, false);

	face_net_.setInput(blob_image, "data");
	cv::Mat detection = face_net_.forward("detection_out");

	const int x_padding = 40;
	const int y_padding = 80;
	cv::Mat detection_mat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
	//阈值为0.5 超过0.5才会显示
	float confidence_threshold = 0.5;
	for (int i = 0; i < detection_mat.rows; i++) {
		float confidence = detection_mat.at<float>(i, 2);
		if (confidence > confidence_threshold) {
			size_t objIndex = (size_t)(detection_mat.at<float>(i, 1));
			float tl_x = detection_mat.at<float>(i, 3) * src_.cols;
			float tl_y = detection_mat.at<float>(i, 4) * src_.rows;
			float br_x = detection_mat.at<float>(i, 5) * src_.cols;
			float br_y = detection_mat.at<float>(i, 6) * src_.rows;
			//原始ROI
			rect_face_ = cv::Rect((int)tl_x, (int)tl_y, (int)(br_x - tl_x), (int)(br_y - tl_y));
			if (rect_face_.area() < 50)
				return false;
			//由于有时候会产生十分奇怪的坐标，故对坐标进行规范化
			if (rect_face_.x > src_.cols || rect_face_.x < 0 || rect_face_.y > src_.rows || rect_face_.x < 0)
			{
				return false;
			}
			//放大后的ROI
			cv::Rect roi;
			roi.x = max(0, rect_face_.x - x_padding);
			roi.y = max(0, rect_face_.y - 4 * y_padding);

			roi.width = rect_face_.width + 2 * x_padding;
			if (roi.width + roi.x > src_.cols - 1)
				roi.width = src_.cols - 1 - roi.x;

			roi.height = rect_face_.height + 2 * y_padding;
			if (roi.height + roi.y > src_.rows - 1)
				roi.height = src_.rows - 1 - roi.y;

			roi_face_all_ = src_(roi);



			return true;
		}

	}
	return false;
}

//采用haar特征提取目标特征位置
/*
* input: 输入图像
* objects_rects：识别出来的目标位置，以方框表示
* min_target_nums: 最少需要识别出来的特征数量，少于这个数，程序将直接跳出
*/
bool FaceGrabber::ObjectDetectHaar(const Mat& input, std::vector<Rect>& objects_rects, size_t min_target_nums)
{
	objects_rects.clear();
	if (input.empty())
		return false;
	std::vector<Rect> parts;
	haar_detector.detectMultiScale(input, parts, 1.2, 6, 0, cv::Size(30, 30));
	if (parts.size() != min_target_nums)
	{
		cout << "cant detect any objects_rects! " << endl;
		return false;
	}
	Mat tmp = input.clone();
	for (int i = 0; i < parts.size(); i++)
	{
		Rect ROI_haar_;
		//添加偏置
		ROI_haar_.x = max(parts[static_cast<int>(i)].x + 10, 0);
		ROI_haar_.y = max(parts[static_cast<int>(i)].y + 10, 0);
		ROI_haar_.width = min(parts[static_cast<int>(i)].width - 20, src_.cols);
		ROI_haar_.height = min(parts[static_cast<int>(i)].height - 20, src_.rows);
		cv::rectangle(tmp, ROI_haar_, cv::Scalar(0, 255, 0), 1, 8, 0);
		objects_rects.push_back(ROI_haar_);
	}
	//cv::imshow("eyes", tmp);

	return false;
}

//语义分割
bool FaceGrabber::FaceDetectTorch(const Mat& input)
{//判空
	if (input.empty())
		return false;
	Mat image_transformed;
	const int set_size = 224;//网络需要的固定图片长宽大小
	const int multiple = 127; // 转换的倍数大小
	//重设尺寸
	resize(input, image_transformed, Size(set_size, set_size));
	cvtColor(image_transformed, image_transformed, COLOR_BGR2RGB);

	// 3.图像转换为Tensor
	torch::Tensor tensor_image = torch::from_blob(image_transformed.data, { image_transformed.rows, image_transformed.cols,3 }, torch::kByte);
	tensor_image = tensor_image.permute({ 2,0,1 });
	tensor_image = tensor_image.toType(torch::kFloat);
	tensor_image = tensor_image.div(255);
	tensor_image = tensor_image.unsqueeze(0);


	//网络前向计算
	torch::Tensor out_tensor_all = sematic_module_.forward({ tensor_image }).toTensor();
	torch::Tensor out_tensor = out_tensor_all.argmax(1);
	out_tensor = out_tensor.squeeze();

	//mul函数，表示张量中每个元素乘与一个数，clamp表示夹紧，限制在一个范围内输出
	//由于一共就三种标签0 1 2， 所以最终mat输出应该是 0 127 254
	out_tensor = out_tensor.mul(multiple).to(torch::kU8);
	out_tensor = out_tensor.to(torch::kCPU);

	dst_torch_.create(set_size, set_size, CV_8U);
	memcpy((void*)dst_torch_.data, out_tensor.data_ptr(), sizeof(torch::kU8) * out_tensor.numel());

	//resize回原来的大小
	resize(dst_torch_, dst_torch_, Size(input.cols, input.rows), 0.0, 0.0, INTER_NEAREST);

	return true;
}
bool FaceGrabber::GetLip(const cv::Mat& input)
{
	if (roi_face_only_.empty())
	{
		cout << "error in GetLip: roi_face_only empty" << endl;
		return false;
	}
	roi_lips_only_.create(Size(input.cols, input.rows), CV_8UC1);
	for (int y = 0; y < input.rows; ++y)
	{
		for (int x = 0; x < input.cols; ++x)
		{
			const Vec3b& origin_pixel = input.at<Vec3b>(y, x);
			//转换为YIQ空间
			const auto& b = (double)origin_pixel[0], g = (double)origin_pixel[1], r = (double)origin_pixel[2];
			const auto  Y = 0.299 * r + 0.587 * g + 0.114 * b;
			const auto  I = 0.596 * r - 0.275 * g - 0.321 * b;
			const auto  Q = 0.212 * r - 0.523 * g + 0.311 * b;
			//进行阈值判断
			if ((Y >= 80 && Y <= 220 && I >= 12 && I <= 78 && Q >= 7 && Q <= 25))
			{
				roi_lips_only_.at<uchar>(y, x) = 255;
			}
			else
			{
				roi_lips_only_.at<uchar>(y, x) = 0;
			}
		}
	}
	Mat dst;
	//对图像进行闭操作
	Mat element = getStructuringElement(MORPH_RECT, Size(10, 15));
	//闭操作
	morphologyEx(roi_lips_only_, roi_lips_only_, MORPH_CLOSE, element);

	cvtColor(roi_lips_only_, dst, COLOR_GRAY2BGR);

	bitwise_and(dst, roi_face_only_, dst);
	cv::imshow("dst", dst);

}
//获得一张有脸和头发的和一张只有脸的 并去除背景
bool FaceGrabber::GetSegments(std::vector<cv::Point>& landmarks)
{
	if (roi_face_all_.empty() || dst_torch_.empty())
		return false;

	const Mat back_ground_mono = Mat::zeros(dst_torch_.size(), CV_8UC1);
	const Mat back_ground_colored = Mat::zeros(dst_torch_.size(), CV_8UC3);

	//配置内存
	//创建一个图像矩阵的矩阵体，之后该图像只有脸
	roi_face_only_ = back_ground_colored.clone();
	mask_face_only_ = back_ground_mono.clone();
	//创建一个图像矩阵的矩阵体，之后该图像只有头发和脸
	roi_face_hair_ = back_ground_colored.clone();
	mask_face_hair_ = back_ground_mono.clone();
	//创建一个图像，之后该图像只有头发
	roi_hair_only_ = back_ground_colored.clone();
	mask_hair_only_ = back_ground_mono.clone();
	//将语义分割的mask分开

	//循环 遍历每个像素
	for (int i = 0; i < dst_torch_.rows; ++i)
	{
		for (int j = 0; j < dst_torch_.cols; ++j)
		{
			auto cur_pixel = dst_torch_.at<uchar>(i, j);
			//如果监测到头发的颜色，有头及脸的图像不做改动，另一张去除头发保持只有脸
			if (cur_pixel == TypeIndex::HAIR)
			{
				//roi_face_hair_.at<Vec3b>(i, j) = roi_face_all_.at<Vec3b>(i, j);
				//mask_face_hair_.at<uchar>(i, j) = 255;

				//roi_hair_only_.at<Vec3b>(i, j) = roi_face_all_.at<Vec3b>(i, j);
				mask_hair_only_.at<uchar>(i, j) = 255;
			}
			//如果监测到脸的颜色，两张图像都保存脸的部分
			else if (cur_pixel == TypeIndex::FACE)
			{
				//roi_face_only_.at<Vec3b>(i, j) = roi_face_all_.at<Vec3b>(i, j);
				mask_face_only_.at<uchar>(i, j) = 255;

				//roi_face_hair_.at<Vec3b>(i, j) = roi_face_all_.at<Vec3b>(i, j);
				//mask_face_hair_.at<uchar>(i, j) = 255;
			}

		}
	}

	//将头发和脸加起来得到roi_face_hair_和roi_face_only_


	//获得dlib脸的mask_actual_face_

	//1.获得外边缘点
	vector<Point> external_edge(landmarks.begin(), landmarks.begin() + 26);
	reverse(external_edge.rbegin(), external_edge.rbegin() + 9);
	//画出maks
	mask_actual_face_ = back_ground_mono.clone();
	fillPoly(mask_actual_face_, external_edge, Scalar(255));
	//简化版bald_head_mask,两颧骨最外侧往上拉个矩形
	mask_bald_head_ = mask_actual_face_.clone();
	//矩形起始点为左侧颧骨往上方拉一个padding_y
	int padding_y = SimpleMath::GetLineLen(landmarks[27], landmarks[33]) * 1.5;
	Point Start(landmarks[0].x, landmarks[0].y - padding_y);
	face_width = SimpleMath::GetLineLen(landmarks[0], landmarks[16]);
	Rect up_padding_rect(Start.x, Start.y, face_width, padding_y);
	rectangle(mask_bald_head_, up_padding_rect, Scalar(255), -1);


	//上鼻子到下嘴唇的距离
	double short_axis = SimpleMath::GetLineLen(landmarks[27], landmarks[8]) * 0.5;
	double long_axis = SimpleMath::GetLineLen(landmarks[0], landmarks[16]) * 1.0;
	//矩形上边的中点为椭圆中点
	auto c_center = Point(Start.x + face_width / 2, Start.y);
	ellipse(mask_bald_head_, c_center, Size(long_axis / 2, short_axis / 2), 0, 180, 360, Scalar(255), -1);
	imshow("ellipse", mask_bald_head_);

	//优化roi_face_切掉脖子区域
	Mat up_padding_mask = back_ground_mono.clone();
	Mat down_padding_mask = back_ground_mono.clone();
	rectangle(up_padding_mask, Point(0, 0), Point(dst_torch_.cols - 1, landmarks[13].y), Scalar(255), -1);
	rectangle(down_padding_mask, Point(0, landmarks[13].y), Point(dst_torch_.cols - 1, dst_torch_.rows - 1), Scalar(255), -1);
	Mat tmp1 = back_ground_mono.clone();
	Mat tmp2 = back_ground_mono.clone();
	mask_face_only_.copyTo(tmp1, up_padding_mask);
	mask_actual_face_.copyTo(tmp2, down_padding_mask);
	mask_face_only_ = tmp1 + tmp2;
	mask_face_hair_ = mask_face_only_ + mask_hair_only_;

	roi_face_all_.copyTo(roi_face_only_, mask_face_only_);
	roi_face_all_.copyTo(roi_hair_only_, mask_hair_only_);

	roi_face_hair_ = roi_hair_only_ + roi_face_only_;

	//眼镜+嘴巴
	if (mask_eyes_only_.empty() || mask_lips_only_.empty())
		return false;
	roi_face_only_.copyTo(roi_eyes_only_, mask_eyes_only_);
	roi_face_only_.copyTo(roi_lips_only_, mask_lips_only_);



	return true;
}



//性别识别
bool FaceGrabber::ProcessGender(const cv::Mat& input)
{
	if (input.empty())
		return false;
	cv::String gender_list[] = { "f", "m" };

	Mat tmp = input.clone();
	resize(tmp, tmp, Size(92, 112));
	cvtColor(tmp, tmp, COLOR_BGR2GRAY);
	equalizeHist(tmp, tmp);
	int classidx = gender_model->predict(tmp);

	cv::String gender = gender_list[classidx];

	cout << gender << endl;
	server_mes_ = gender;
	return true;
}

bool FaceGrabber::IsFaceStable(const vector<Point>& land_marks)
{
	static deque<Point> eyes_center;
	server_mes_ = "not yet";

	//判读人是否正脸看着屏幕
	//眼睛中点离脸轮廓中点的位置

	const double dir_thres1 = 15.0;
	auto midpt = SimpleMath::GetMidpt(land_marks[0], land_marks[16]);
	double mid_error = SimpleMath::GetLineLen(midpt, land_marks[27]);
	cout << "mid deviation" << mid_error << endl;


	//判断眼睛连线是否平行
	const double thres = 15.0;

	const Point2d ref_vec(1, 0);
	Point2d left_eye = objects_eyes[0].tl();
	Point2d right_eye = objects_eyes[1].tl();
	objects_eyes.clear();
	Point2d dir_vec = right_eye - left_eye;

	double deviation = SimpleMath::GetVectorAngleDegree(ref_vec.x, ref_vec.y, dir_vec.x, dir_vec.y);
	deviation = fabs(deviation);
	cout << "deviation: " << deviation << endl;
	if (deviation > thres || mid_error > dir_thres1)
	{
		//修改信息
		eyes_center.clear();
		server_mes_ = "adjust face";
		cout << "error adjust face degree" << endl;
		return false;
	}

	//先存个左眼位置，如果左眼位置稳定，就进行下一步操作

	if (eyes_center.size() < 5)
	{
		eyes_center.push_back(left_eye);
		return false;
	}
	else
	{
		eyes_center.pop_front();
		eyes_center.push_back(left_eye);
		double err = 0.0;
		for (int i = 0; i < eyes_center.size() - 1; ++i)
		{
			err += SimpleMath::GetLineLen(eyes_center[i], eyes_center[i + 1]);
		}
		if (err < 50)
		{
			eyes_center.clear();
			return true;
		}
		else
			return false;
	}

	return false;

}


//对闭运算  消除黑线
void FaceGrabber::MorphologyClose(cv::Mat& img, const int& kernel_size)
{
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(kernel_size, kernel_size));
	morphologyEx(img, img, MORPH_CLOSE, kernel);
	//medianBlur(img, img, 3);
}


void FaceGrabber::CleanDisk(const std::string& path)
{
	const string suffix(".png");
	remove((path + string("head-2") + suffix).c_str());
	remove((path + string("face-2") + suffix).c_str());
	remove((path + string("head-1") + suffix).c_str());
}

//图片居中测试
void FitFaceToMask(Size& mask_size, Mat& input, Mat& dst, double face_width)
{
	Mat output = Mat::zeros(mask_size, CV_8UC4);

	int i_width = mask_size.width;

	double ratio = 0.5 * i_width / face_width;

	Mat expanded_input;
	resize(input, expanded_input, Size(), ratio, ratio);
	cout << "expand_input size " << expanded_input.size() << endl;

	int new_x = mask_size.width / 2 - expanded_input.cols / 2;
	int new_y = mask_size.height - expanded_input.rows;
	Rect new_rect(new_x, new_y, expanded_input.cols, expanded_input.rows);

	expanded_input.copyTo(output(new_rect));


	imshow("output", output);
	output.copyTo(dst);
}


void FaceGrabber::WritePic2Disk(const std::string& path)
{
	CleanDisk(path);


	//face_hair
	/*MorphologyClose(mask_face_hair_, 10);
	roi_face_all_.copyTo(roi_face_hair_, mask_face_hair_);
	PhotoShop::SmoothEdges(roi_face_hair_, mask_face_hair_, roi_face_hair_);*/

	double ratio = 0;
	int output_size = 250;
	PhotoShop::RemoveBackground(bald_head_, mask_bald_head_, bald_head_, true);
	cout << face_width << endl;
	FitFaceToMask(Size(output_size, output_size), bald_head_, bald_head_, face_width);
	PhotoShop::RemoveBackground(roi_face_hair_, mask_face_hair_, roi_face_hair_, true);
	FitFaceToMask(Size(output_size, output_size), roi_face_hair_, roi_face_hair_, face_width);
	waitKey(1);
	//cvtColor(roi_face_hair_, roi_face_hair_, COLOR_BGR2BGRA);
	//resize(roi_face_hair_, roi_face_hair_, Size(200, 240), 0.0, 0.0, 0);

	/*mask_bald_head_ = Mat::zeros(bald_head_.size(), CV_8UC1);
	PhotoShop::GetFilledMask(bald_head_, mask_bald_head_, false);
	MorphologyClose(mask_bald_head_, 10);
	roi_face_all_.copyTo(bald_head_, mask_bald_head_);*/

	/*cout << "bald_head size" << bald_head_.size() << endl;
	cout << "mask_bald_size" << mask_bald_head_.size() << endl;*/

	//cvtColor(bald_head_, bald_head_, COLOR_BGR2BGRA);

	//resize(bald_head_, bald_head_, Size(200, 240));

	//imwrite(path + string("face-all.png"), roi_face_all_);
	resize(roi_face_all_, roi_face_all_, Size(450, 450), 0.0, 0.0, 0);
	Mat mask(roi_face_all_.size(), roi_face_all_.type(), Scalar(0, 0, 0, 0));
	circle(mask, Point(200, 200), 200, Scalar(255, 255, 255, 255), -1);
	bitwise_and(roi_face_all_, mask, roi_face_all_);


	PhotoShop::RemoveBackground(roi_eyes_only_, mask_eyes_only_, roi_eyes_only_, false);

	if (!server_mes_.empty())
	{
		const string suffix(".png");
		imwrite(path + string("head-2") + suffix, roi_face_hair_);
		imwrite(path + string("face-2") + suffix, bald_head_);
		imwrite(path + string("head-1") + suffix, roi_face_all_);
		imwrite(path + string("eyes-1") + suffix, roi_eyes_only_);
	}
}

//Scalar FaceGrabber::GetSkinColor(const cv::Mat& input)
//{
//	if (input.channels() != 3)
//		return Scalar();
//
//	Mat dst = input.clone();
//
//	for (auto& rect : objects_eyes)
//	{
//		ZoomRect(rect, 10, 10, input.size());
//		cv::rectangle(dst, rect, Scalar(0, 0, 0), -1);
//	}
//	cv::imshow("dst", dst);
//
//	size_t color_r = 0, color_g = 0, color_b = 0, pix_size = 0;
//
//	for (int x = 0; x < dst.cols; ++x)
//	{
//		for (int y = 0; y < dst.rows; ++y)
//		{
//			const Vec3b& pixel_color = dst.at<Vec3b>(y, x);
//			if (pixel_color[0] != 0 && pixel_color[1] != 0 && pixel_color[2] != 0)
//			{
//				color_r += pixel_color[0];
//				color_b += pixel_color[1];
//				color_b += pixel_color[2];
//				++pix_size;
//			}
//		}
//	}
//	return Scalar(color_r / pix_size, color_g / pix_size, color_b / pix_size);
//}

void FaceGrabber::GetSkinColor(cv::Mat& input)
{
	if (input.empty())
	{
		cout << "input is empty" << endl;
		return;
	}

	int num_colored_pixels = 0;
	int a_blue = 0, a_green = 0, a_red = 0;

	for (int x = 0; x < input.cols; ++x)
	{
		for (int y = 0; y < input.rows; ++y)
		{
			const Vec3b& cur_pixel = input.at<Vec3b>(y, x);
			if (cur_pixel[0] != 0)
			{
				a_blue += cur_pixel[0];
				a_green += cur_pixel[1];
				a_red += cur_pixel[2];
				++num_colored_pixels;
			}
		}
	}

	a_blue /= num_colored_pixels;
	a_green /= num_colored_pixels;
	a_red /= num_colored_pixels;
	skin_color_ = Scalar(a_blue, a_green, a_red);
}

void FaceGrabber::ZoomRect(cv::Rect& rect, const int x, const int y, cv::Size pic_size)
{
	rect.x = max(0, rect.x - x);
	rect.y = max(0, rect.y - y);
	rect.width = rect.x + 2 * x;
	if (rect.width > pic_size.width)
		rect.width = pic_size.width - 1 - rect.x;
	if (rect.height > pic_size.height)
		rect.height = pic_size.height - 1 - rect.y;
}

void FaceGrabber::RotateFace(std::vector<cv::Point>& land_marks, Mat& input)
{
	if (land_marks.empty() || input.empty())
		return;

	////歪脸修正
	Point face_dir = land_marks[33] - land_marks[27];
	cout << face_dir << endl;
	double angle = SimpleMath::GetVectorAngleDegree(0, 90, face_dir.x, face_dir.y);
	cout << "current angle " << angle << endl;
	if (face_dir.x > 0)
	{
		angle = -angle;

	}

	int w = input.cols;
	int h = input.rows;
	Point center(w / 2, h / 2);
	//获取仿射变换矩阵
	Mat rotation = getRotationMatrix2D(center, angle, 1);
	//仿射变换
	warpAffine(input, input, rotation, roi_face_all_.size());
	return;
}



void FaceGrabber::Release()
{
	server_mes_.clear();
	objects_eyes.clear();
}

//显示相机一帧帧图像
void FaceGrabber::ShowSrc()
{
	if (src_.empty())
		return;
	cv::imshow("src", src_);
	waitKey(1);
}
//显示语义分割的图像
void FaceGrabber::ShowDstTorch()
{
	cv::imshow("dst_torched", dst_torch_);


	return;
}
//显示一张有脸和头发的和一张只有脸的
void FaceGrabber::ShowROIFace()
{
	cv::imshow("roi_face", roi_face_all_);
	cv::imshow("roi_face_hair_", roi_face_hair_);
	cv::imshow("roi_face_only_", roi_face_only_);

}


bool FaceGrabber::GetFacialLandMarks(const cv::Mat& input, std::vector<cv::Point>& land_marks)
{
	if (input.empty())
	{
		cout << "error empty input on GetFaceLandMarks" << endl;
		return false;
	}
	land_marks.clear();
	dlib::cv_image<dlib::bgr_pixel> cimg(input);

	//检测人脸，得到roi方框
	vector<dlib::rectangle> faces = dlib_detector_(cimg);

	if (faces.empty())
	{
		cout << "no face detected" << endl;
		return false;
	}
	//找到landmarks
	dlib::full_object_detection current_pos = pose_model_(cimg, faces[0]);

	if (current_pos.num_parts() == 68)
	{
		for (int i = 0; i < 68; ++i)
		{
			land_marks.emplace_back(current_pos.part(i).x(), current_pos.part(i).y());
		}
	}
	else
	{
		cout << "wrong number in landmark detection" << endl;
		return false;
	}
}

bool FaceGrabber::GetFacialFeatures(cv::Mat& input, const std::vector<cv::Point>& land_marks)
{
	if (input.empty() || land_marks.empty())
	{
		server_mes_ = "error in Facial Features, empty input";
		cout << server_mes_ << endl;
		return false;
	}


	mask_eyes_only_ = cv::Mat::zeros(input.size(), CV_8UC1);
	mask_lips_only_ = cv::Mat::zeros(input.size(), CV_8UC1);

	const int area_ratio = 30;
	auto f = [area_ratio](RotatedRect& rr)
	{
		if (rr.boundingRect().area() < area_ratio)
			return false;
		return true;
	};

	vector<Point> left_eye_points, right_eye_points, lips_points;
	//将左眼区域进行连接
	for (int i = 36; i < 41; i++)
	{
		cv::line(mask_eyes_only_, cvPoint(land_marks[i].x, land_marks[i].y), cvPoint(land_marks[i + 1].x, land_marks[i + 1].y), Scalar(255), 1);
		left_eye_points.emplace_back(land_marks[i].x, land_marks[i].y);
	}
	cv::line(mask_eyes_only_, cvPoint(land_marks[36].x, land_marks[36].y), cvPoint(land_marks[41].x, land_marks[41].y), Scalar(255), 1);



	auto rrec1 = minAreaRect(left_eye_points);
	if (rrec1.boundingRect().area() < area_ratio)
		return false;
	objects_eyes.push_back(rrec1.boundingRect());

	//将右眼区域进行连接
	for (int i = 42; i < 47; i++)
	{
		cv::line(mask_eyes_only_, cvPoint(land_marks[i].x, land_marks[i].y), cvPoint(land_marks[i + 1].x, land_marks[i + 1].y), Scalar(255), 1);
		right_eye_points.emplace_back(land_marks[i].x, land_marks[i].y);
	}
	cv::line(mask_eyes_only_, cvPoint(land_marks[42].x, land_marks[42].y), cvPoint(land_marks[47].x, land_marks[47].y), Scalar(255), 1);
	auto rrec2 = minAreaRect(right_eye_points);
	if (rrec2.boundingRect().area() < area_ratio)
		return false;
	objects_eyes.push_back(rrec2.boundingRect());

	//将嘴巴区域进行连接
	for (int i = 48; i < 60; i++)
	{
		cv::line(mask_lips_only_, cvPoint(land_marks[i].x, land_marks[i].y), cvPoint(land_marks[i + 1].x, land_marks[i + 1].y), Scalar(255), 1);
		lips_points.emplace_back(land_marks[i].x, land_marks[i].y);
	}
	cv::line(mask_lips_only_, cvPoint(land_marks[48].x, land_marks[48].y), cvPoint(land_marks[60].x, land_marks[60].y), Scalar(255), 1);
	auto rrec3 = minAreaRect(lips_points);
	if (rrec3.boundingRect().area() < area_ratio)
		return false;

	Point nose_tip(land_marks[30].x, land_marks[30].y);
	//取肤色
	skin_color_ = input.at<Vec3b>(nose_tip.y, nose_tip.x);

	//PhotoShop::FillContour(mask_lips_only_, mask_lips_only_, 255);
	//PhotoShop::FillContour(mask_eyes_only_, mask_eyes_only_, 255);

	PhotoShop::GetFilledMask(mask_eyes_only_, mask_eyes_only_, true);
	PhotoShop::GetFilledMask(mask_lips_only_, mask_lips_only_, false);

	return true;

}

void FaceGrabber::ShowBaldHead()
{
	cv::imshow("bald_head", bald_head_);

	waitKey(1);
}

void FaceGrabber::ShowDebug()
{
	cv::imshow("lips", roi_lips_only_);
	cv::imshow("eyes", roi_eyes_only_);
	waitKey(1);
}

PicParser_PT GetFaceParser()
{
	return &FaceGrabber::GetInstance();
}

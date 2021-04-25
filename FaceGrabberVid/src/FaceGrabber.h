#pragma once
#include "../config.h"
#include "../PS_Mixer.h"
#include "../IPicParser.h"



const cv::Scalar BODY_COLOR{ 143, 172, 229 };

typedef IPicParser* PicParser_PT;

class FaceGrabber : public IPicParser
{
public:
	//相机控制
	bool StarGrab();
	void GetFrame();

	static FaceGrabber& GetInstance()
	{
		static FaceGrabber f1;
		return f1;
	}
	//继承的接口实现
	void SetSrc(const std::string& input)
	{
		src_ = cv::imread(input);
	}

	//总特征识别
	bool ProcessFace();

	//识别性别
	std::string GetFrameResult() { return server_mes_; }

	//释放所有的资源
	void Release();

	//输出调试结果
	void ShowSrc();
	void ShowDstTorch();
	void ShowROIFace();
	void ShowBaldHead();
	void ShowDebug();

	void CleanDisk(const std::string& path);
	void WritePic2Disk(const std::string& path);

	//存储3个部分颜色的结构体
	enum TypeIndex
	{
		BACKGROUND = 0,
		FACE = 127,
		HAIR = 254,
		LIPS = 129,
		EYES = 128
	};
private:
	//屏蔽的方法

	//默认构造函数
	inline FaceGrabber()
	{
		//读入各个模型
		if (!haar_detector.load(haar_file_name))
		{
			std::cout << "error loading haar_file !" << std::endl;
		}
		//torch模型
		sematic_module_ = torch::jit::load(torch_file_name);
		//opencv脸模型
		face_net_ = cv::dnn::readNetFromTensorflow(model_bin, config_text);
		face_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
		face_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
		//性别模型
		//gender_net_ = cv::dnn::readNet(genderModel, genderProto);
		//gender_model = cv::face::EigenFaceRecognizer::create();
		gender_model = cv::Algorithm::load<cv::face::EigenFaceRecognizer>(eigen_net);

		//五官模型
		dlib::deserialize(DlibModel) >> pose_model_;
		dlib_detector_ = dlib::get_frontal_face_detector();

		//获取美颜实例
		xiu_ptr = XIU_Beautify::GetInstance();

	}
	~FaceGrabber()
	{
		xiu_ptr->Release();
	}
	FaceGrabber(const FaceGrabber&);
	FaceGrabber& operator= (const FaceGrabber&);

private:
	//方法

	//特征识别 
	//哈尔特征识别
	bool ObjectDetectHaar(const cv::Mat& input, std::vector< cv::Rect >& objects_rects, size_t min_target_nums);
	//深度学习获取人脸区域
	bool GetFace();
	//使用torch库对人脸进行像素级语义分割
	bool FaceDetectTorch(const cv::Mat& input);
	//获得嘴唇区域掩膜, 掩膜为单通道图像
	bool GetLip(const cv::Mat& input);

	//dlib 应用
	//获取面部所有的landmarks
	bool GetFacialLandMarks(const cv::Mat& input, std::vector<cv::Point>& land_marks);
	//获取五官信息
	bool GetFacialFeatures(cv::Mat& input, const std::vector<cv::Point>& land_marks);

	//根据torch库的分割效果，对图像进行切割
	bool GetSegments(std::vector<cv::Point>& landmarks);

	//计算性别
	bool ProcessGender(const cv::Mat& input);

	//判断脸是否不动
	bool IsFaceStable(const std::vector<cv::Point>& land_marks);

	//闭操作
	void MorphologyClose(cv::Mat& img, const int& kernel_size);

	//获得皮肤颜色
	void GetSkinColor(cv::Mat& input);
	//根据图片size放大rect
	void ZoomRect(cv::Rect& rect, const int x, const int y, cv::Size pic_size);

	//对脸部进行旋转操作
	void RotateFace(std::vector<cv::Point>& land_marks, cv::Mat& input);


private:
	//分类器
	cv::CascadeClassifier haar_detector;

	torch::jit::Module sematic_module_;

	cv::dnn::Net face_net_;
	cv::dnn::Net gender_net_;

	dlib::shape_predictor pose_model_;
	dlib::frontal_face_detector dlib_detector_;

	cv::Ptr<cv::face::BasicFaceRecognizer> gender_model;

	//视频控制器
	cv::VideoCapture cap_;
	//Mat
	cv::Mat src_;
	cv::Mat dst_;
	cv::Mat dst_torch_;

	//美颜器
	XIU_Beautify* xiu_ptr;

	//face_beautified
	cv::Mat face_beautified_;

	//返回给客户端的信息
	std::string server_mes_;


	//脸宽
	int face_width;

	//输出到客户端的Mat以及各自的单通道255mask
	//未经过裁切 脸+头发+背景
	cv::Mat roi_face_all_;
	//脸加头发
	cv::Mat roi_face_hair_;
	cv::Mat mask_face_hair_;
	//只有脸
	cv::Mat roi_face_only_;
	cv::Mat mask_face_only_;
	//只有头发
	cv::Mat roi_hair_only_;
	cv::Mat mask_hair_only_;
	//只有嘴唇
	cv::Mat roi_lips_only_;
	cv::Mat mask_lips_only_;
	//只有眼睛
	cv::Mat roi_eyes_only_;
	cv::Mat mask_eyes_only_;
	//dlib生成的精确脸部mask
	cv::Mat mask_actual_face_;


	//秃头
	cv::Mat bald_head_;
	cv::Mat mask_bald_head_;

	//脸部方框
	cv::Rect rect_face_;

	//眼部roi
	std::vector<cv::Rect> objects_eyes;

	//皮肤色彩
	cv::Scalar skin_color_;

};

PicParser_PT GetFaceParser();

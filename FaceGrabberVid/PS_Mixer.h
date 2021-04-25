#pragma once
#include "src/opencv_includes.h"
#include "src/std_includes.h"
#include "config.h"
#include "inc/XIUSDK_BeautyEngine.h"

/*
第一次写工厂类
我们尝试对不同的图层混合算法进行测试，直到找到我们需要的类
*/

namespace MIX_TYPE
{
	const std::string COLOR("COLOR");
}

//定义抽象类
class FaceBeautify
{
public:
	virtual bool Beautify(const cv::Mat src, cv::Mat output) = 0;
	virtual bool Release() = 0;
	virtual ~FaceBeautify() {}
protected:
	void* handle_;
};

class XIU_Beautify : public FaceBeautify
{
public:
	XIU_Beautify()
	{
		auto GetProgramDir = []()
		{
			char exeFullPath[MAX_PATH]; // Full path   
			std::string strPath = "";

			GetModuleFileNameA(NULL, exeFullPath, MAX_PATH);

			strPath = (std::string)exeFullPath;    // Get full path of the file   

			return strPath.substr(0, strPath.find_last_of('\\'));
		};

		handle_ = (BeautyHandle)Beauty_InitBeautyEngine(GetProgramDir().c_str());
	}

	static XIU_Beautify* GetInstance();
	bool  Beautify(const cv::Mat src, cv::Mat output);
	bool  Release();

private:
	XIU_Beautify(const XIU_Beautify&);
	XIU_Beautify& operator= (const XIU_Beautify&);
};





////定义各个实现类
////颜色混合模式
//class ColorMixer : public PS_Mixer
//{
//public:
//	bool Mix(const cv::Mat& src, const cv::Mat& mask, cv::Mat& output);
//	bool Release();
//};

class PhotoShop
{
public:
	//填充闭合轮廓
	static void FillContour(cv::Mat& input, cv::Mat& output, const uchar mask_val = 255);
	//获得彩图的mask, 输入必须是三通道,返回一张255的mask, 标志位决定了是否需要添加
	static void GetFilledMask(cv::Mat& input, cv::Mat& mask, bool all_drawn);

	//总处理函数
	static void FaceBeautify(cv::Mat& input, cv::Mat& output);
	/*
	dx ,fc 磨皮程度与细节程度的确定 双边滤波参数
	transparency 透明度
	*/
	static void FaceGrinding(cv::Mat& input, cv::Mat& output, int value1 = 3, int value2 = 1);//磨皮
	//saturation    max_increment
	static void AdjustSaturation(cv::Mat& input, cv::Mat& output, int saturation = 0, const int max_increment = 200);
	//alpha 调整对比度				beta 调整亮度
	static void AdjustBrightness(cv::Mat& input, cv::Mat& output, float alpha = 1.1, float beta = 40);
	//覆盖图层函数
	static void ApplyMask(const std::string& mask_type, const cv::Mat& input, const cv::Mat& mask, cv::Mat& dst);

	//获取皮肤颜色

	//补全光头
	static bool GetBaldHead(cv::Mat input, cv::Mat input_mask, cv::Mat output_mask, cv::Mat& dst);

	//去除背景
	static void RemoveBackground(cv::Mat& img, cv::Mat& mask, cv::Mat& dst, bool resize_to_fit = true);

	//平滑边缘
	static void SmoothEdges(cv::Mat& img, cv::Mat& mask, cv::Mat& dst);
};

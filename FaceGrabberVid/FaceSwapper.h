#pragma once
#include "IPicParser.h"
#include "dlib_includes.h"
#include "config.h"

class FaceSwapper /*: public IPicParser*/
{
public:
	static FaceSwapper& GetInstance()
	{
		static FaceSwapper f1;
		return f1;
	}
	void SetSrc(const std::string& input);
	void SetSTDImg(const std::string& input);
	void SetSTDImg(const cv::Mat& input);
	bool ProcessFace();
	std::string GetFrameResult();
	void WritePic2Disk();
private:
	//设置为单例模式
	FaceSwapper()
	{
		InitDlib();
	}
	FaceSwapper(const FaceSwapper&);
	FaceSwapper& operator=(const FaceSwapper&);

	//初始化方法
	void InitDlib();
	//方法
	bool GetFaceLandMarks(const cv::Mat& input, std::vector<cv::Point>& land_marks);
	// Calculate Delaunay triangles for set of points
	// Returns the vector of indices of 3 points for each triangle
	void CalculateDelaunayTriangles(cv::Rect rect, std::vector<cv::Point2f>& points, std::vector<std::vector<int>>& delaunary_tri);
	// Warps and alpha blends triangular regions from img1 and img2 to img
	void warpTriangle(cv::Mat& img1, cv::Mat& img2, std::vector<cv::Point2f>& t1, std::vector<cv::Point2f>& t2);

	// Apply affine transform calculated using srcTri and dstTri to src
	void applyAffineTransform(cv::Mat& warpImage, cv::Mat& src, std::vector<cv::Point2f>& srcTri, std::vector<cv::Point2f>& dstTri);


private:
	//分类器
	dlib::shape_predictor pose_model_;
	dlib::frontal_face_detector dlib_detector_;

	//各个Mat
	cv::Mat src_;
	//标准图像
	cv::Mat std_img_;

	//存放src的脸部特征点
	std::vector<cv::Point> src_face_marks_;
	//存放标准的脸部特征点
	std::vector<cv::Point> std_face_marks_;

};

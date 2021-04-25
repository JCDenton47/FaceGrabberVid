#include "FaceSwapper.h"

using namespace std;
using namespace cv;

void FaceSwapper::SetSrc(const std::string& input)
{
	src_ = imread(input);
	if (input.empty())
	{
		cout << "error empty file" << endl;
	}
}

void FaceSwapper::SetSTDImg(const std::string& input)
{
	std_img_ = imread(input);
}

void FaceSwapper::SetSTDImg(const cv::Mat& input)
{
	std_img_ = input.clone();
}

bool FaceSwapper::ProcessFace()
{

	if (!GetFaceLandMarks(std_img_, std_face_marks_) || !GetFaceLandMarks(src_, src_face_marks_))
	{
		cout << "error loading face imgs" << endl;
		return false;
	}


	Mat img_wraped = std_img_.clone();
	src_.convertTo(src_, CV_32F);
	img_wraped.convertTo(img_wraped, CV_32F);

	//find convex_hull
	vector<Point2f> hull1;
	vector<Point2f> hull2;

	vector<int> hull_index;

	convexHull(std_face_marks_, hull_index, false, false);

	for (int i = 0; i < hull_index.size(); ++i)
	{
		hull1.push_back(src_face_marks_[hull_index[i]]);
		hull2.push_back(std_face_marks_[hull_index[i]]);
	}

	//三角变换
	vector<vector<int>> dt;
	Rect rect(0, 0, img_wraped.cols, img_wraped.rows);
	CalculateDelaunayTriangles(rect, hull2, dt);

	// Apply affine transformation to Delaunay triangles
	for (size_t i = 0; i < dt.size(); i++)
	{
		vector<Point2f> t1, t2;
		// Get points for img1, img2 corresponding to the triangles
		for (size_t j = 0; j < 3; j++)
		{
			t1.push_back(hull1[dt[i][j]]);
			t2.push_back(hull2[dt[i][j]]);
		}


		warpTriangle(src_, img_wraped, t1, t2);

	}

	// Calculate mask
	vector<Point> hull8U;
	for (int i = 0; i < hull2.size(); i++)
	{
		Point pt(hull2[i].x, hull2[i].y);
		hull8U.push_back(pt);
	}

	Mat mask = Mat::zeros(std_img_.rows, std_img_.cols, std_img_.depth());
	fillConvexPoly(mask, &hull8U[0], hull8U.size(), Scalar(255, 255, 255));

	// Clone seamlessly.
	Rect r = boundingRect(hull2);
	Point center = (r.tl() + r.br()) / 2;

	Mat output;
	img_wraped.convertTo(img_wraped, CV_8UC3);
	seamlessClone(img_wraped, std_img_, mask, center, output, NORMAL_CLONE);
	imshow("Face Swapped", output);
	waitKey(1);
}

void FaceSwapper::InitDlib()
{
	dlib::deserialize(DlibModel) >> pose_model_;
	dlib_detector_ = dlib::get_frontal_face_detector();
}

bool FaceSwapper::GetFaceLandMarks(const cv::Mat& input, std::vector<cv::Point>& land_marks)
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

void FaceSwapper::CalculateDelaunayTriangles(cv::Rect rect, std::vector<cv::Point2f>& points, std::vector<std::vector<int>>& delaunary_tri)
{
	// Create an instance of Subdiv2D
	Subdiv2D subdiv(rect);

	// Insert points into subdiv
	for (vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
		subdiv.insert(*it);

	vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	vector<Point2f> pt(3);
	vector<int> ind(3);

	for (size_t i = 0; i < triangleList.size(); i++)
	{
		Vec6f t = triangleList[i];
		pt[0] = Point2f(t[0], t[1]);
		pt[1] = Point2f(t[2], t[3]);
		pt[2] = Point2f(t[4], t[5]);

		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])) {
			for (int j = 0; j < 3; j++)
				for (size_t k = 0; k < points.size(); k++)
					if (abs(pt[j].x - points[k].x) < 1.0 && abs(pt[j].y - points[k].y) < 1)
						ind[j] = k;

			delaunary_tri.push_back(ind);
		}
	}


}

void FaceSwapper::warpTriangle(cv::Mat& img1, cv::Mat& img2, std::vector<cv::Point2f>& t1, std::vector<cv::Point2f>& t2)
{

	Rect r1 = boundingRect(t1);
	Rect r2 = boundingRect(t2);

	// Offset points by left top corner of the respective rectangles
	vector<Point2f> t1Rect, t2Rect;
	vector<Point> t2RectInt;
	for (int i = 0; i < 3; i++)
	{

		t1Rect.push_back(Point2f(t1[i].x - r1.x, t1[i].y - r1.y));
		t2Rect.push_back(Point2f(t2[i].x - r2.x, t2[i].y - r2.y));
		t2RectInt.push_back(Point(t2[i].x - r2.x, t2[i].y - r2.y)); // for fillConvexPoly

	}

	// Get mask by filling triangle
	Mat mask = Mat::zeros(r2.height, r2.width, CV_32FC3);
	fillConvexPoly(mask, t2RectInt, Scalar(1.0, 1.0, 1.0), 16, 0);

	// Apply warpImage to small rectangular patches
	Mat img1Rect;
	img1(r1).copyTo(img1Rect);

	Mat img2Rect = Mat::zeros(r2.height, r2.width, img1Rect.type());

	applyAffineTransform(img2Rect, img1Rect, t1Rect, t2Rect);

	multiply(img2Rect, mask, img2Rect);
	multiply(img2(r2), Scalar(1.0, 1.0, 1.0) - mask, img2(r2));
	img2(r2) = img2(r2) + img2Rect;


}

void FaceSwapper::applyAffineTransform(cv::Mat& warpImage, cv::Mat& src, std::vector<cv::Point2f>& srcTri, std::vector<cv::Point2f>& dstTri)
{
	// Given a pair of triangles, find the affine transform.
	Mat warpMat = getAffineTransform(srcTri, dstTri);

	// Apply the Affine Transform just found to the src image
	warpAffine(src, warpImage, warpMat, warpImage.size(), INTER_LINEAR, BORDER_REFLECT_101);
}

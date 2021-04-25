#pragma once
#include "src/std_includes.h"
#include "src/opencv_includes.h"
#include "src/torch_lib_includes.h"
#include "src/dlib_includes.h"
#include "src/SimpleMath.h"
#include <direct.h>

const std::string haar_file_name("Resource_Depo/face/haarcascade_eye_tree_eyeglasses.xml");
const std::string torch_file_name("Resource_Depo/face/Face_sematic_seg_model.pt");
const cv::String model_bin = "Resource_Depo/face/opencv_face_detector_uint8.pb";
const cv::String config_text = "Resource_Depo/face/opencv_face_detector.pbtxt";
const cv::String genderProto = "Resource_Depo/gender/gender_deploy.prototxt";
const cv::String genderModel = "Resource_Depo/gender/gender_net.caffemodel";
const cv::String eigen_net = "Resource_Depo/gender/eigen.yml";
const cv::String DlibModel = "Resource_Depo/dlib/shape_predictor_68_face_landmarks.dat";

const cv::String OutputPath = "E:\\UnityProjects\\Beauty\\Assets\\Resources\\";

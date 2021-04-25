#include <iostream>
#include "src/MesManagerFactory.h"
#include "src/FaceGrabber.h"
#include "FaceSwapper.h"
#include "IPicParser.h"
#include "src/WzSerialPort.h"

using namespace std;
using namespace cv;

//int main()
//{
//	PicParser_PT pt = GetFaceParser();
//	MesManagerFactory mm_factory;
//	MM_PTR socket_manager = mm_factory.GetInstance("TCP");
//	WzSerialPort light_switch;
//
//	while (!socket_manager->ServerStart())
//	{
//		cout << "retry" << endl;
//	}
//
//	//执行死循环
//	while (socket_manager->ProcessMes())
//	{
//		string mes = socket_manager->GetReceivedMessage();
//		if (mes.empty())
//		{
//			cout << "receive a empty message!" << endl;
//			socket_manager->WaitNewConnect();
//		}
//		else if (mes.find("png") != string::npos)
//		{
//
//			light_switch.OpenLight();
//			waitKey(60);
//			pt->SetSrc(mes);
//			string output_path = mes.substr(0, mes.find_last_of("/") + 1);
//			if (pt->ProcessFace())
//			{
//
//				const string result = pt->GetFrameResult();
//
//				cout << "进去" << endl;
//				//	light_switch.CloseLight();
//				//	Sleep(100);
//				if (result == "m" || result == "f")
//				{
//					waitKey(1);
//					pt->WritePic2Disk(output_path);
//					pt->Release();
//					light_switch.CloseLight();
//					socket_manager->SendPack(result.c_str(), result.size());
//					socket_manager->WaitNewConnect();
//
//				}
//
//			}
//			else
//			{
//				const string result = pt->GetFrameResult();
//				socket_manager->SendPack(result.c_str(), result.size());
//
//			}
//			//light_switch.CloseLight();
//			waitKey(60);
//
//		}
//
//
//	}
//
//	return 1;
//}





//int main()
//{
//	WzSerialPort w;
//
//	while (true)
//	{
//		w.OpenLight();
//		Sleep(1000);
//		w.CloseLight();
//		Sleep(1000);
//	}
//
//
//}

int main()
{
	FaceGrabber* pt = &FaceGrabber::GetInstance();
	pt->StarGrab();
	//执行死循环
	while (true)
	{
<<<<<<< HEAD
		pt->GetFrame();
		pt->ProcessFace();
=======
		string mes = socket_manager->GetReceivedMessage();
		if (mes.empty())
			continue;
		else if (mes.find("png") != string::npos)
		{
			pt->SetSrc(mes);
			pt->ProcessFace();
			if (!pt->GetGender().empty())
			{
				socket_manager->SendPack(pt->GetGender().c_str(), pt->GetGender().size());
			}
			else
			{
				const string ping("not yet\n");
				socket_manager->SendPack(ping.c_str(), ping.size());
			}
		}
>>>>>>> 7197924... 添加了应答机制，防止了Unity因为无应答而卡死在阻塞处

		waitKey(1);
		/*pt->ShowDebug();
		pt->ShowBaldHead();
		pt->WritePic2Disk();*/
	}

	return 1;
}


//int main()
//{
//	MesManagerFactory mm_factory;
//	MM_PTR socket_manager = mm_factory.GetInstance("TCP");
//
//	socket_manager->ServerStart();
//
//	while (socket_manager->ProcessMes())
//	{
//
//	}
//}

//int main()
//{
//	Mat img1 = imread("1.png");
//	Mat img2 = imread("2.png");
//
//	cout << img1.channels() << endl;
//	cout << img2.channels() << endl;
//
//	FaceSwapper* fs = &FaceSwapper::GetInstance();
//
//	VideoCapture cap;
//	cap.open(0);
//	if (!cap.isOpened())
//	{
//		return -1;
//	}
//	while (true)
//	{
//		Mat src;
//		cap >> src;
//		src.convertTo(src, CV_8UC3);
//		fs->SetSrc("2.png");
//		fs->SetSTDImg(src);
//		fs->ProcessFace();
//	}
//
//	//fs->SetSTDImg("1.png");
//
//	//fs->ProcessFace();
//}



//int main(int argc, char** argv)
//{
//	Mat src = imread("head-2.png");
//	imshow("src", src);
//	waitKey();
//}

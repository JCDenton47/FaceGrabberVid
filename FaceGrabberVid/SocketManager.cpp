#include "SocketManager.h"

using namespace std;

bool SocketManager::InitManager()
{
	server_ = INVALID_SOCKET;
	int result = 0;
	result = WSAStartup(MAKEWORD(2, 2), &data_);
	//出现接收错误
	if (result != 0)
	{
		cout << " WSAStartup() init error " << GetLastError() << endl;
		return false;
	}
	//服务器接口属性定义
	server_ = socket(AF_INET, SOCK_STREAM, 0);

	addr_srv_.sin_family = AF_INET;
	addr_srv_.sin_port = htons(1080);//转换为网络字节序，小端序
	addr_srv_.sin_addr.S_un.S_addr = htonl(INADDR_ANY);//ip port 
	//将socket和地址绑定起来
	result = ::bind(server_, (LPSOCKADDR)&addr_srv_, sizeof(SOCKADDR_IN));
	if (result != 0)
	{
		std::cout << "winsock bind() error" << result;
		system("pause");
		return false;
	}

	return true;
}

bool SocketManager::ServerStart()
{
	int result = 0;

	if (server_ == INVALID_SOCKET)
	{
		cout << "server start error, init server first !" << endl;
		return false;
	}

	result = listen(server_, 1000);

	if (result)
	{
		cout << "server listen error" << endl;
		return false;
	}

	WaitNewConnect();

}

bool SocketManager::ProcessMes()
{

	char* f = NULL;


	//收信息
	char recvBuff[1024];
	memset(recvBuff, 0, sizeof(recvBuff));

	ReceivePack(recvBuff, sizeof(recvBuff));

	string message(recvBuff);
	received_message_ = message;
	if (message.empty())
		return true;
	else if (message.find("png") != string::npos)
	{
	}
	std::cout << "packet size" << strlen(recvBuff) << std::endl;
	std::cout << "recv from client:" << recvBuff << std::endl;


	return true;
}

bool SocketManager::ServerStop()
{
	closesocket(server_);
	WSACleanup();
	return true;
}

bool SocketManager::SendPack(const char* mes, int len)
{
	const int bufsize = 1024;
	char buf[bufsize];
	if (len > bufsize)
		return false;
	memset(buf, 0, bufsize);
	memcpy(buf, mes, len);
	int iSend = send(socket_conn_, buf, bufsize, 0);
	if (iSend == SOCKET_ERROR)
	{
		std::cout << "send error error";
		return false;
	}
}

bool SocketManager::ReceivePack(char* mes, int len)
{
	int i_recv = recv(socket_conn_, mes, len, 0);
	if (i_recv == SOCKET_ERROR)
	{
		cout << "receive error" << endl;
		return false;
	}
	return true;
}

bool SocketManager::WaitNewConnect()
{
	int len = sizeof(SOCKADDR);

	cout << "wait new connect..." << endl;
	//给server链接上真正的客户端
	socket_conn_ = accept(server_, (SOCKADDR*)&addr_client_, &len);

	if (socket_conn_ == SOCKET_ERROR)
	{
		std::cout << " accept error" << WSAGetLastError();
		return false;
	}
	return true;
}

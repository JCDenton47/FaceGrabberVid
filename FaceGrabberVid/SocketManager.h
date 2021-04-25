#pragma once

#include <WinSock2.h>

#include "IPicParser.h"
#include "src/IMessageManager.h"
#include "config.h"

//引入socket库
#pragma comment(lib, "ws2_32.lib")


//实现一个tcp/ip客户端
class SocketManager : public IMessageManager
{
public:

	//获取实例
	static SocketManager& getInstance()
	{
		static SocketManager instance;
		return instance;
	}
	//开始监听
	bool ServerStart();
	//处理收发信息
	bool ProcessMes();
	//关闭服务器
	bool ServerStop();
	//发送信息
	bool SendPack(const char* mes, int len);
	//接受信息
	bool ReceivePack(char* mes, int len);
	//接受新的监听
	bool WaitNewConnect();

	std::string GetReceivedMessage()
	{
		return received_message_;
	}

private:
	//初始化
	bool InitManager();

	SocketManager()
	{
		InitManager();
	}
	~SocketManager() {}
	SocketManager(const SocketManager&);
	SocketManager& operator= (const SocketManager&);
private:

	SOCKET server_;//socket object
	WSADATA data_;//存储socket属性的结构体
	SOCKADDR_IN addr_srv_; //定义了服务端可以接受哪些ip和port的访问
	SOCKADDR_IN addr_client_;//存储连接好的服务端的地址

	SOCKET socket_conn_;

	std::string received_message_;
};

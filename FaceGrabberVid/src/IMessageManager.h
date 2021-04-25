#pragma once
#include <string>

class IMessageManager
{
public:
	//开始监听
	virtual bool ServerStart() = 0;
	//处理收发信息
	virtual bool ProcessMes() = 0;
	//关闭服务器
	virtual bool ServerStop() = 0;
	//发送信息	 
	virtual bool SendPack(const char* mes, int len) = 0;
	//接受信息
	virtual bool ReceivePack(char* mes, int len) = 0;
	virtual std::string GetReceivedMessage() = 0;

	//设置为等待新连接状态	
	virtual bool WaitNewConnect() = 0;

protected:
	//初始化
	virtual bool  InitManager() = 0;


};

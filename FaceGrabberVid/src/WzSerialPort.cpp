
#include "WzSerialPort.h"




void WzSerialPort::OpenLight()
{
	if (IsLightOn)
		return;
	open("COM3", 9600, 0, 8, 1);
	send("open", 10);
	std::cout << "light opened" << std::endl;
	IsLightOn = true;
}
void WzSerialPort::CloseLight()
{
	if (!IsLightOn)
		return;
	open("COM3", 9600, 0, 8, 1);
	send("close", 10);
	std::cout << "light off" << std::endl;

	IsLightOn = false;

}
WzSerialPort::WzSerialPort()
{
	IsLightOn = false;


}

WzSerialPort::~WzSerialPort()
{

}

bool WzSerialPort::open(const char* portname,
	int baudrate,
	char parity,
	char databit,
	char stopbit,
	char synchronizeflag)
{
	this->synchronizeflag = synchronizeflag;
	HANDLE hCom = NULL;
	if (this->synchronizeflag)
	{
		//同步方式
		hCom = CreateFileA(portname, //串口名
			GENERIC_READ | GENERIC_WRITE, //支持读写
			0, //独占方式，串口不支持共享
			NULL,//安全属性指针，默认值为NULL
			OPEN_EXISTING, //打开现有的串口文件
			0, //0：同步方式，FILE_FLAG_OVERLAPPED：异步方式
			NULL);//用于复制文件句柄，默认值为NULL，对串口而言该参数必须置为NULL
	}
	else
	{
		//异步方式
		hCom = CreateFileA(portname, //串口名
			GENERIC_READ | GENERIC_WRITE, //支持读写
			0, //独占方式，串口不支持共享
			NULL,//安全属性指针，默认值为NULL
			OPEN_EXISTING, //打开现有的串口文件
			FILE_FLAG_OVERLAPPED, //0：同步方式，FILE_FLAG_OVERLAPPED：异步方式
			NULL);//用于复制文件句柄，默认值为NULL，对串口而言该参数必须置为NULL
	}

	if (hCom == (HANDLE)-1)
	{
		return false;
	}

	//配置缓冲区大小 
	if (!SetupComm(hCom, 1024, 1024))
	{
		return false;
	}

	// 配置参数 
	DCB p;
	memset(&p, 0, sizeof(p));
	p.DCBlength = sizeof(p);
	p.BaudRate = baudrate; // 波特率
	p.ByteSize = databit; // 数据位

	switch (parity) //校验位
	{
	case 0:
		p.Parity = NOPARITY; //无校验
		break;
	case 1:
		p.Parity = ODDPARITY; //奇校验
		break;
	case 2:
		p.Parity = EVENPARITY; //偶校验
		break;
	case 3:
		p.Parity = MARKPARITY; //标记校验
		break;
	}

	switch (stopbit) //停止位
	{
	case 1:
		p.StopBits = ONESTOPBIT; //1位停止位
		break;
	case 2:
		p.StopBits = TWOSTOPBITS; //2位停止位
		break;
	case 3:
		p.StopBits = ONE5STOPBITS; //1.5位停止位
		break;
	}

	if (!SetCommState(hCom, &p))
	{
		// 设置参数失败
		return false;
	}

	//超时处理,单位：毫秒
	//总超时＝时间系数×读或写的字符数＋时间常量
	COMMTIMEOUTS TimeOuts;
	TimeOuts.ReadIntervalTimeout = 1000; //读间隔超时
	TimeOuts.ReadTotalTimeoutMultiplier = 500; //读时间系数
	TimeOuts.ReadTotalTimeoutConstant = 5000; //读时间常量
	TimeOuts.WriteTotalTimeoutMultiplier = 500; // 写时间系数
	TimeOuts.WriteTotalTimeoutConstant = 2000; //写时间常量
	SetCommTimeouts(hCom, &TimeOuts);

	PurgeComm(hCom, PURGE_TXCLEAR | PURGE_RXCLEAR);//清空串口缓冲区

	memcpy(pHandle, &hCom, sizeof(hCom));// 保存句柄

	return true;
}

void WzSerialPort::close()
{
	HANDLE hCom = *(HANDLE*)pHandle;
	CloseHandle(hCom);
}

int WzSerialPort::send(const void* buf, int len)
{
	HANDLE hCom = *(HANDLE*)pHandle;

	if (this->synchronizeflag)
	{
		// 同步方式
		DWORD dwBytesWrite = len; //成功写入的数据字节数
		BOOL bWriteStat = WriteFile(hCom, //串口句柄
			buf, //数据首地址
			dwBytesWrite, //要发送的数据字节数
			&dwBytesWrite, //DWORD*，用来接收返回成功发送的数据字节数
			NULL); //NULL为同步发送，OVERLAPPED*为异步发送
		if (!bWriteStat)
		{
			return 0;
		}
		return dwBytesWrite;
	}
	else
	{
		//异步方式
		DWORD dwBytesWrite = len; //成功写入的数据字节数
		DWORD dwErrorFlags; //错误标志
		COMSTAT comStat; //通讯状态
		OVERLAPPED m_osWrite; //异步输入输出结构体

		//创建一个用于OVERLAPPED的事件处理，不会真正用到，但系统要求这么做
		memset(&m_osWrite, 0, sizeof(m_osWrite));
		//m_osWrite.hEvent = CreateEvent(NULL, TRUE, FALSE, "WriteEvent");

		ClearCommError(hCom, &dwErrorFlags, &comStat); //清除通讯错误，获得设备当前状态
		BOOL bWriteStat = WriteFile(hCom, //串口句柄
			buf, //数据首地址
			dwBytesWrite, //要发送的数据字节数
			&dwBytesWrite, //DWORD*，用来接收返回成功发送的数据字节数
			&m_osWrite); //NULL为同步发送，OVERLAPPED*为异步发送
		if (!bWriteStat)
		{
			if (GetLastError() == ERROR_IO_PENDING) //如果串口正在写入
			{
				WaitForSingleObject(m_osWrite.hEvent, 1000); //等待写入事件1秒钟
			}
			else
			{
				ClearCommError(hCom, &dwErrorFlags, &comStat); //清除通讯错误
				CloseHandle(m_osWrite.hEvent); //关闭并释放hEvent内存
				return 0;
			}
		}
		return dwBytesWrite;
	}
}

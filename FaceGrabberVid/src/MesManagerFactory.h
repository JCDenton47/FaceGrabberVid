#pragma once
#include "IMessageManager.h"
#include "../SocketManager.h"
#include <memory>


typedef std::shared_ptr<IMessageManager> MM_PTR;


struct MesManagerFactory
{

	MM_PTR GetInstance(const std::string& input)
	{
		if (input == "TCP")
		{
			return MM_PTR(&SocketManager::getInstance(), std::mem_fun(&SocketManager::ServerStop));
		}
	}
};

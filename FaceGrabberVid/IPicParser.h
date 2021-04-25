#pragma once
#include <string>
#include <memory>

struct IPicParser
{
	virtual void SetSrc(const std::string& input) = 0;
	virtual bool ProcessFace() = 0;
	virtual std::string GetFrameResult() = 0;
	virtual void WritePic2Disk(const std::string& path) = 0;
	virtual void Release() = 0;
};

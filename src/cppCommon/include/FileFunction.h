#pragma once
#include <fstream>
#include <sstream>
//#include <direct.h>
#if WIN32 || _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
#include "StringFunction.h"
//文件名
inline void toStandardPath(std::string& vioPath);//转换成标准路径：即用正斜杠'/'而不是用反斜杠'\'
inline std::string extractFilePath(const std::string& vFullFileName);
inline std::string extractFileName(const std::string& vFullFileName);
template<class Functor> size_t readPathOrText(const std::string& vImagePathOrFileName, std::vector<std::string>& voFileNameSet, Functor pred, size_t vOffset=0, bool vWithPath=true);
inline std::string assembleFullFileName(const std::string& vPath, const std::string& vFileName)
{
	if (vPath.empty()) return vFileName;
	return (vPath.back()=='/' || vPath.back()=='\\')? vPath+vFileName : vPath + "/" + vFileName;
}
inline bool isExist(const std::string& vFileOrDirectory)
{
#if WIN32 || _WIN32
	return 0 == _access(vFileOrDirectory.c_str(), 0);
#else
	return 0 == access(vFileOrDirectory.c_str(), 0);
#endif
}

//扩展名
inline std::string deleteExtentionName(const std::string& vFileName, bool vDotReserving=true, char vDot='.');
inline std::string extractExtentionName(const std::string& vFileName);

//读写二进制文件
inline bool writeFromMemory2File(const std::string& vFileName, const void* vpBinaryData, size_t vBites);
inline bool readBinaryDataFromFile2Memory(const std::string& vFileName, void* &vopBuffer, size_t& vioBites);

template<typename TStream>
bool openFileStream(const std::string& vFileName, std::ios_base::openmode vMode, TStream& vioFStream, const std::string& vErrInfor)
{
	vioFStream.open(vFileName, vMode);
	if (!vioFStream)
	{
		std::cout << vErrInfor << std::endl;
		return false;
	}
	return true;
}


template<typename TParam>
void inputeParameter(TParam& voValue, int argc, char **argv, const std::string& vPromptMessage, int vIndex=1)
{
	if (vIndex+1 > argc)
	{
		std::cout << vPromptMessage << std::endl;
		std::cin >> voValue;
		return;
	}
	std::stringstream stream(argv[vIndex]);
	stream >> voValue;
}

void toStandardPath(std::string& vioPath)
{
	char* pTemp = (char*) vioPath.c_str();
	while (*pTemp)
	{
		if ('\\' == *pTemp)
			*pTemp = '/';
		pTemp++;
	}
}


std::string extractFilePath(const std::string& vFullFileName)
{
	int iSeprator = vFullFileName.rfind('/');
	if (0 > iSeprator)
		iSeprator = vFullFileName.rfind('\\');
	iSeprator++;
	return vFullFileName.substr(0, iSeprator);
}

inline std::string extractFileName(const std::string& vFullFileName)
{
	int iSeprator = vFullFileName.rfind('/');
	if (0 > iSeprator)
		iSeprator = vFullFileName.rfind('\\');
	iSeprator++;
	return vFullFileName.substr(iSeprator);
}

template<class Functor>// bool (*pFileFilter)(const std::string& vFileName); [](const std::string& v){return 
size_t readPathOrText(const std::string& vImagePathOrFileName, std::vector<std::string>& voFileNameSet, Functor pred, size_t vOffset, bool vWithPath)
{
	if (!isExist(vImagePathOrFileName))
		std::cout << "Fail to read " << vImagePathOrFileName << std::endl;

	std::string NamesFile = vImagePathOrFileName;
	int dotOffset = vImagePathOrFileName.rfind('.');
	if (dotOffset<0 || dotOffset>vImagePathOrFileName.size())
	{
			NamesFile = ".filenameset";
		std::string CMD = "ls ";
	#if WIN32 || _WIN32
		CMD = "dir /b ";
	#endif
		CMD += vImagePathOrFileName + " >" + NamesFile;
		system(CMD.c_str());
	}

	std::ifstream Fin;
	if (openFileStream(NamesFile, std::ios::in, Fin, "Can't read directory of " + vImagePathOrFileName))
	{
		Fin.seekg(0, std::ios::end);
		size_t offset = Fin.tellg();
		Fin.seekg(vOffset, std::ios::beg);
		vOffset = offset;
		std::string FileName;
		while (std::getline(Fin, FileName))
		{
			while (FileName.back() == '\n' || FileName.back() == '\r' || FileName.back() == ' ')
				FileName.pop_back();
			if (FileName == NamesFile)
				continue;
			if (pred(FileName))
				voFileNameSet.push_back(vWithPath ? assembleFullFileName(vImagePathOrFileName, FileName) : FileName);
		}
		Fin.close();
	}

	return vOffset;
}

bool writeFromMemory2File(const std::string& vFileName, const void* vpBinaryData, size_t vBites)
{
	std::ofstream Fout(vFileName, std::ios::binary);
	if (Fout.fail()) { std::cout << "Fail to wirte data to file " << vFileName << std::endl; ; return false; }
	Fout.write((const char*)vpBinaryData, vBites);
	Fout.close();
	return true;
}


bool readBinaryDataFromFile2Memory(const std::string& vFileName, void* &vopBuffer, size_t& vioBites)
{
	std::ifstream Fin(vFileName, std::ios::binary);
	if (Fin.fail()) { std::cout << "Fail to read data from file " << vFileName << std::endl; ; return false; }
	if (vioBites < 1)
	{
		Fin.seekg(0, std::ios_base::end);
		vioBites = Fin.tellg();
		Fin.seekg(0, std::ios_base::beg);
	}
	if (NULL == vopBuffer) vopBuffer = malloc(vioBites);
	Fin.read((char*)vopBuffer, vioBites);
	Fin.close();
	return true;
}


std::string deleteExtentionName(const std::string& vFileName, bool vDotReserving, char vDot)
{
	char temp = 0;
	auto pBegin = vFileName.crbegin(), pEnd = vFileName.crend();
	for (; pBegin != pEnd; pBegin++)
	{
		temp = *pBegin;
		if (temp == vDot || temp=='\\' || temp=='/') break;
	}
	std::string fileName = vFileName.substr(0, temp == vDot ? (pEnd-pBegin-1):vFileName.size());
	if (vDotReserving)
		fileName.push_back(vDot);
	return fileName;
}

std::string extractExtentionName(const std::string& vFileName)
{
	int IndexOfDot = vFileName.rfind('.');
	if (IndexOfDot < 0) return "";
	return vFileName.substr(IndexOfDot+1);
}

//文件夹：新建文件夹_mkdir，如果存在的话不会删除再建立

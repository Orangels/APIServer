#pragma once
#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <string.h> 
//#include <windows.h>

//大小写转换
inline std::string& toUpper(std::string& vioStr);
inline std::string toUpperCopy(const std::string& vStr);
inline std::string& toLower(std::string& vioStr);
inline std::string toLowerCopy(const std::string& vStr);

//消除左右空格
inline std::string& ltrim(std::string& vioStr, char vTemp=' ');
inline std::string& rtrim(std::string& vioStr, char vTemp=' ');
inline std::string& trim(std::string& vioStr, char vTemp);
inline std::string& trim(std::string& vioStr);

//字符串与其他数据类型转换
template<typename SimpleType> inline std::string convert2String(const SimpleType& vData);
template<typename SimpleType> inline SimpleType convertFromString(const std::string& vData);
template<typename SimpleType> inline void fillArray(const std::string& vValues, SimpleType* vopBase, int vNumElement=-1);
template<typename SimpleType> inline void appandData(const std::string& vValues, std::vector<SimpleType>& voDstContainer, int vNumElement=-1);

// inline void toWideChar(const std::string& vSrc, std::vector<WCHAR>& voDst);//转为宽字符Unicode
inline bool isEndWith(const std::string& vStr, const std::string& vPostfix)
{
	if (vPostfix.length() > vStr.length()) return false;
	return 0 == strcmp(vPostfix.c_str(), vStr.c_str()+vStr.length()-vPostfix.length());
}
inline bool isStartWith(const std::string& vStr, const std::string& vPrefix)
{
	std::string src = vStr;
	ltrim(src, '\r');
	ltrim(src, '\t');
	ltrim(src);
	const char *pPre=vPrefix.c_str(), *pStr=src.c_str();
	while (*pPre)
	{
		if (*pPre++ != *pStr++)
			return false;
	}
	return true;
}
inline void updateProgress(unsigned int vIndex);
inline void splitString(std::string vSrc, std::vector<std::string>& voDst, const char vSeparator, char vIgnoring = ' ', bool vClearBeforAppanding = true)
{
	if (vClearBeforAppanding) voDst.resize(0);
	char* pSrc = (char*)vSrc.c_str(), *pEnd = pSrc + vSrc.size();
	while (*pSrc)
	{
		if (vSeparator == *pSrc || vIgnoring == *pSrc)
			*pSrc = 0;
		pSrc++;
	}

	pSrc = (char*)vSrc.c_str();
	while (pSrc < pEnd)
	{
		if (*pSrc == 0)
			pSrc++;
		else
		{
			voDst.push_back(pSrc);
			pSrc += voDst.back().size();
		}
	}
}
template<typename T>
inline void splitStringAndConvert(std::string vSrc, std::vector<T>& voDst, const char vSeparator, char vIgnoring = ' ', bool vClearBeforAppanding = true)
{
	if (vClearBeforAppanding) voDst.resize(0);
	std::stringstream SS(vSrc);
	T temp;
	char sep;
	while (SS >> temp)
	{
		voDst.push_back(temp);
		SS >> sep;
	}
}

inline int replaceString(std::string& vioStr, const std::string& vSrc, const std::string& vDst, int vOffset=0, bool vRecursion=false)
{
	int Index = vioStr.find(vSrc, vOffset);
	if (Index < 0) return Index;

	int diff = vDst.size() - vSrc.size();
	if (diff > 0)
		vioStr.resize(vioStr.size()+diff);
	if (0 != diff)
		memcpy((char*)vioStr.c_str() + Index + vDst.size(), (char*)vioStr.c_str() + Index + vSrc.size(), vioStr.size()-Index-vSrc.size()-diff);
	memcpy((char*)vioStr.c_str() + Index, vDst.c_str(), vDst.size());
	if (diff < 0)
		vioStr.resize(vioStr.size() + diff);
	if (vRecursion)
		while((Index=replaceString(vioStr, vSrc, vDst, Index + vDst.size())>0));
	return Index;
}






std::string& toUpper(std::string& vioStr)
{
	char *pKey = (char*)vioStr.c_str();
	int Len = vioStr.length();
	while (Len--)
	{
		if (*pKey<='z' && *pKey>='a')
			*pKey -= 32;
		pKey++;
	}
	return vioStr;
}

std::string toUpperCopy(const std::string& vStr)
{
	std::string Str = vStr;
	toUpper(Str);
	return Str;
}

std::string& toLower(std::string& vioStr)
{
	char *pKey = (char*)vioStr.c_str();
	int Len = vioStr.length();
	while (Len--)
	{
		if (*pKey<='Z' && *pKey>='A')
			*pKey += 32;
		pKey++;
	}
	return vioStr;
}

std::string toLowerCopy(const std::string& vStr)
{
	std::string Str = vStr;
	toLower(Str);
	return Str;
}


std::string& ltrim(std::string& vioStr, char vTemp)
{
	char *p1 = (char*)vioStr.c_str(), *p2 = p1;
	while(vTemp == *p2)
		p2++;

	if (p2 > p1)
		vioStr = vioStr.substr(p2 - p1);
	return vioStr;
}

std::string& rtrim(std::string& vioStr, char vTemp)
{
	int NewLen = vioStr.length();
	char *pStr = (char*)vioStr.c_str() + NewLen - 1;
	while(NewLen)
	{
		if (vTemp != *pStr)
			break;
		NewLen--;
		pStr--;
	}
	if (vioStr.length() != NewLen)
		vioStr.resize(NewLen);
	return vioStr;
}

std::string& trim(std::string& vioStr, char vTemp)
{
	ltrim(vioStr, vTemp);
	return rtrim(vioStr, vTemp);
}

inline std::string& trim(std::string& vioStr)
{
	ltrim(vioStr, '\r');
	ltrim(vioStr, '\t');
	ltrim(vioStr);
	rtrim(vioStr, '\n');
	rtrim(vioStr, '\r');
	rtrim(vioStr);
    return vioStr;
}


template<typename SimpleType> std::string convert2String(const SimpleType& vData)
{
	std::stringstream StringConvertor;
	StringConvertor << vData;
	std::string Result;
	StringConvertor >> Result;
	return Result;
}

template<typename SimpleType> SimpleType convertFromString(const std::string& vData)
{
	std::stringstream  StringConvertor(vData);
	SimpleType Temp;
	StringConvertor >> Temp;
	return Temp;
}

template<typename SimpleType> void fillArray(const std::string& vValues, SimpleType* vopBase, int vNumElement)
{
	std::stringstream g_StringConvertor(vValues);
	if (vNumElement > 0)
		while (vNumElement-- && (g_StringConvertor >> *vopBase++));
	else
		while(g_StringConvertor >> *vopBase++);
}

template<typename SimpleType> void appandData(const std::string& vValues, std::vector<SimpleType>& voDstContainer, int vNumElement)
{
	std::stringstream g_StringConvertor(vValues);
	if (vNumElement > 0)
	{
		int Offset = voDstContainer.size();
		voDstContainer.resize(Offset+vNumElement);
		while (vNumElement-- && (g_StringConvertor>>voDstContainer[Offset++]));
	}
	else 
	{
		SimpleType Temp;
		while (g_StringConvertor >> Temp)
			voDstContainer.push_back(Temp);
	}
}

/*
void toWideChar(const std::string& vSrc, std::vector<WCHAR>& voDst)
{
	int NumChar = vSrc.length() + 1;
	if (voDst.size() < NumChar) voDst.resize(NumChar);
	MultiByteToWideChar(CP_ACP, 0, vSrc.c_str(), NumChar, voDst.data(), NumChar);
}
*/


inline void updateProgress(bool vRestart=false, int vBegin=0)
{
	static int Index = 0;
	if (vRestart) Index = vBegin;
	if (Index < 11) std::cout << "\b" << Index;
	else if (Index < 101) std::cout << "\b\b" << Index;
	else if (Index < 1001) std::cout << "\b\b\b" << Index;
	else if (Index < 10001) std::cout << "\b\b\b\b" << Index;
	else if (Index < 100001) std::cout << "\b\b\b\b\b" << Index;
	else if (Index < 1000001) std::cout << "\b\b\b\b\b\b" << Index;
	else if (Index < 10000001) std::cout << "\b\b\b\b\b\b\b" << Index;
	else if (Index < 100000001) std::cout << "\b\b\b\b\b\b\b\b" << Index;
	else if (Index < 1000000001) std::cout << "\b\b\b\b\b\b\b\b\b" << Index;
	else std::cout << "\b\b\b\b\b\b\b\b\b\b" << Index;
	Index++;
}

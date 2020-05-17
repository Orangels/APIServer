#pragma once
#include <string>
#include <vector> 
#include "Singleton.h"

class CPythonCaller: public CSingleton<CPythonCaller>
{
public:
    CPythonCaller();
	virtual ~CPythonCaller();

	float getData(const std::string& vDataFileName, void* vop0Data, void* vop1Data = NULL);
	float getData(const std::string& vDataFileName, std::vector<void*>& vBuffers);
	void getData(const std::string& vDataFileName, std::vector<std::vector<float>>& voBuffers);
	float call(const std::string& vFunctionName, std::vector<std::pair<void*, size_t>>& vBuffers, const std::string& vInfor="");
	float call(const std::string& vFunctionName, std::vector<void*>& vBuffers, std::vector<size_t>& vBitesOfBuffers);
	float call(const std::string& vFunctionName, void* vp0buffer, size_t v0size, void* vp1buffer=NULL, size_t v1size=0);

private:
//	CPythonCaller();
	friend class CSingleton<CPythonCaller>;
	void* __appandSampleParameter(const std::string& vFormat, ...);
	void* __appandArrayParameter(void* vpData, int vBiytes);
	void* m_pPythonFunctions;
};




/*
CMakeLists.txt:
	find_package(PythonLibs)
	include_directories(${PYTHON_INCLUDE_DIRS})
	target_link_libraries(exeOrDynamiclib  ${PYTHON_LIBRARIES})

example:
python:
import os
import sys
import struct
import numpy as np
def add(a,b,c):
	print ("a = " , a, c, type(c)       )
	b = np.frombuffer(b, dtype=c)
	print ("in python function add"  )
	print ("b = " , b      )
	print ("ret = " , a+b[0] , type(b)     )
	b[0] = 0
	b[1] = 65537
	b[2] = 0

	return  a
cpp:
CPythonCaller pyCaller("../pytest.py");
pyCaller.appandSampleParameter(2, "i");
int data[] = { 256,3,4,4 };
pyCaller.appandArrayParameter(data, "i", sizeof(data));
pyCaller.call("add");
pyCaller.appandSampleParameter(333.3f, "f");
pyCaller.appandArrayParameter(data, "i", sizeof(data));
int r = pyCaller.call("add");
std::cout << r << std::endl;
*/
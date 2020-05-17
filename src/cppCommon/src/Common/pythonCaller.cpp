#include <numeric>
#include <iostream>
#include <algorithm>
#include <Python.h> 
#include<stdarg.h>
#include "pythonCaller.h"
#include "FileFunction.h"

CPythonCaller::CPythonCaller()
{
	if (!(isExist("functions4cpp.py") || isExist("../functions4cpp.py")))
	{
		std::cout << "errors: can't find file of functions4cpp.py" << std::endl; exit(0);
	}
	else if (!isExist("functions4cpp.py"))
		system("ln -s ../functions4cpp.py functions4cpp.py");
	if (!Py_IsInitialized())
		Py_Initialize();
	PyRun_SimpleString("import sys");
	std::string path =  "./";
	PyRun_SimpleString(("sys.path.append(\"" + path + "\")").c_str());
	PyObject *pName, *pModule = NULL;
	pName = PyUnicode_FromString("functions4cpp");//PyUnicode_DecodeFSDefault

    if (pName == NULL) {
        PyErr_Print();
        throw std::invalid_argument("Error: PyUnicode_FromString");
    }
	pModule = PyImport_Import(pName);
    if (pModule == NULL) {
        PyErr_Print();
        throw std::invalid_argument("fails to import the module");
    }
	//Py_DECREF(pModule);
	//Py_DECREF(pName);
	m_pPythonFunctions = PyModule_GetDict(pModule);
}

CPythonCaller::~CPythonCaller()
{
	if (Py_IsInitialized())
		Py_Finalize();
}


#define  TYPYConvertor(a) {pPar = Py_BuildValue(vFormat.c_str(), va_arg(arg_ptr, a)); }
void* CPythonCaller::__appandSampleParameter(const std::string& vFormat, ...)
{
	PyObject * pPar = NULL;
	va_list arg_ptr; 
	va_start(arg_ptr, 1);
	if ("c" == vFormat) TYPYConvertor(char)
	else if ("s" == vFormat) TYPYConvertor(char*)
	else if ("i" == vFormat) TYPYConvertor(int)
	else if ("f" == vFormat) TYPYConvertor(float)
	else std::cout << "You need add a kind of format in the function of CPythonCaller::appandSampleParameter in the file of pythonCaller.cpp !!" << std::endl;
	va_end(arg_ptr);
	return pPar;
}


void* CPythonCaller::__appandArrayParameter(void* vpData, int vBiytes)
{
    if (vpData==0 || vBiytes==0)
        return __appandSampleParameter("i", 0);
	Py_buffer *buf = (Py_buffer *)malloc(sizeof(*buf));
	int r = PyBuffer_FillInfo(buf, NULL, vpData, vBiytes, 0, PyBUF_CONTIG);
	PyObject *mv = PyMemoryView_FromBuffer(buf);
	free(buf);
	return mv;
}


float CPythonCaller::call(const std::string& vFunctionName, void* vp0buffer, size_t v0size, void* vp1buffer/*=NULL*/, size_t v1size/*=0*/)
{
	std::vector<std::pair<void*, size_t>> buffer2(2, std::make_pair(vp0buffer, v0size));
	if (NULL == vp1buffer || 0 == v1size) buffer2.pop_back();
	else
	{
		buffer2.back().first = vp1buffer;
		buffer2.back().second = v1size;
	}
	return call(vFunctionName, buffer2);
}

float CPythonCaller::call(const std::string& vFunctionName, std::vector<void*>& vBuffers, std::vector<size_t>& vBitesOfBuffers)
{
	int n = std::min(vBitesOfBuffers.size(), vBuffers.size());
	std::vector<std::pair<void*, size_t>> bufferes;
	bufferes.reserve(n);
	for (int i = 0; i < n; ++i)
		bufferes.emplace_back(std::make_pair(vBuffers[i], vBitesOfBuffers[i]));
	return call(vFunctionName, bufferes);
}


float CPythonCaller::call(const std::string& vFunctionName, std::vector<std::pair<void*, size_t>>& vBuffers, const std::string& vInfor)
{
	PyObject * pPythonFunctions = (PyObject *)m_pPythonFunctions;
	PyObject* pFunc = PyDict_GetItemString(pPythonFunctions, vFunctionName.c_str());
	PyObject *pArgs; *pArgs;
	if ("" != vInfor)
	{
		pArgs = PyTuple_New(vBuffers.size()+1);
		PyTuple_SetItem(pArgs, 0, (PyObject *)__appandSampleParameter("s", vInfor.c_str()));
		for (int i=0; i<vBuffers.size(); ++i)
			PyTuple_SetItem(pArgs, i+1, (PyObject *)__appandArrayParameter(vBuffers[i].first, vBuffers[i].second));
	}
	else
	{
		pArgs = PyTuple_New(vBuffers.size());
		for (int i = 0; i < vBuffers.size(); ++i)
			PyTuple_SetItem(pArgs, i, (PyObject *)__appandArrayParameter(vBuffers[i].first, vBuffers[i].second));
	}

	PyObject* pReturn = PyObject_CallObject(pFunc, pArgs);

	float result = 0;
	PyArg_Parse(pReturn, "f", &result);

	return result;
}

float CPythonCaller::getData(const std::string& vDataFileName, std::vector<void*>& vBuffers)
{
	std::vector<std::pair<void*, size_t>> bufferes(1);
	bufferes.back() = std::make_pair(vBuffers.data(), vBuffers.size() * sizeof(void*));
	return call("getData", bufferes, vDataFileName);
}

float CPythonCaller::getData(const std::string& vDataFileName, void* vop0Data, void* vop1Data /*= NULL*/)
{
	std::vector<void*> pointers(2, vop0Data);
	if (NULL == vop1Data) pointers.pop_back();
	else pointers.back() = vop1Data;
	return getData(vDataFileName, pointers);
}


void CPythonCaller::getData(const std::string& vDataFileName, std::vector<std::vector<float>>& voBuffers)
{
	std::vector<void*> buffer(8);
	getData(vDataFileName, buffer);
	int* pNum = (int*)buffer.front();
	int num = *pNum++;
	voBuffers.resize(num);
	for (int i = 0; i < num; ++i)
	{
		voBuffers[i] = std::vector<float>(*pNum++);
		memcpy(voBuffers[i].data(), buffer[i + 1], voBuffers[i].size() * 4);
	}
}

extern "C"
{
	void memcpfromPython2cpp(void* src, void* dst, size_t vBites)
	{
// 		int n = vBites / 4;
// 		double fsum = 0; float* p = (float*)src;
// 		while (n--) { fsum += *p < 0 ? -*p : *p; p++; }
// 		std::cout << vBites/4 << " cpp sum(abs(data)): " << fsum << std::endl;
		memcpy(dst, src, vBites);
	}
}

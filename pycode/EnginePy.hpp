#ifndef __ENGINEPY_HPP__
#define __ENGINEPY_HPP__

#include <Python.h>
#include <stdio.h>
#include <time.h>
#include <opencv2/core.hpp>
#include <numpy/arrayobject.h>

#define LOG_DEBUG(msg, ...) printf("[%s][%s][%s][%d]: " msg, __TIME__, __FILE__, __FUNCTION__, __LINE__, ##__VA_ARGS__)

using namespace std;
using namespace cv;

class Engine_api
{
public:
    PyObject* m_pDict = NULL;
    PyObject* m_pHandle = NULL;
public:
    Engine_api();
    Engine_api(std::string pyClass);
    ~Engine_api();
    vector<int> get_result(Mat frame, std::string mode);
    void test();
};

#endif
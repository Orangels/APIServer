#include <Python.h>
#include <numpy/arrayobject.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>

using namespace std;

void mat2np(cv::Mat img, PyObject* ArgList, uchar *CArrays);

void vec2np(vector<int> arr, PyObject* vArgList, int singleLen, int* CArrays);
void vec2np(vector<float> arr, PyObject* vArgList, int singleLen, float * CArrays);
void vec2np(PyObject* vArgList, int size, int singleLen, float* CArrays);


void list2vector(PyObject* pyResult,std::vector<int> &vret);
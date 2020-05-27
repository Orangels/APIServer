#include <iostream>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "utils/mat2numpy.h"


int init_numpy2()
{
    import_array();
}

void mat2np(cv::Mat img, PyObject* vArgList, uchar *CArrays){
    
    init_numpy2();
    auto sz = img.size();
    int x = sz.width;
    int y = sz.height;
    int z = img.channels();
//    uchar *CArrays = new uchar[x*y*z];//这一行申请的内存需要释放指针，否则存在内存泄漏的问题
    int iChannels = img.channels();
    int iRows = img.rows;
    int iCols = img.cols * iChannels;

    if (img.isContinuous())
    {
        iCols *= iRows;
        iRows = 1;
    }
    CArrays = (uchar*)img.data;

    npy_intp Dims[3] = { y, x, z}; //注意这个维度数据！
    PyObject *PyArray = PyArray_SimpleNewFromData(3, Dims, NPY_UBYTE, CArrays);
    PyTuple_SetItem(vArgList, 0, PyArray);
//    delete CArrays ;

}

void list2vector(PyObject* pyResult,std::vector<int> &vret){

        int list_len = PyObject_Size(pyResult);
        PyObject *pRet = NULL;
        int ret;//标志位
        for (int i = 0; i < list_len; i++)
        {
            pRet = PyList_GetItem(pyResult, i);//根据下标取出python列表中的元素
            PyArg_Parse(pRet, "i", &ret);
            vret.push_back(ret);
        }
}
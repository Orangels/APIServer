#include "EnginePy.hpp"
#include "utils/mat2numpy.h"
#include <iostream>

Engine_api::Engine_api()
{
    PyObject* pFile = NULL;
    PyObject* pModule = NULL;
    PyObject* pClass = NULL;

    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure(); //申请获取GIL
    Py_BEGIN_ALLOW_THREADS;
    Py_BLOCK_THREADS;

    do
    {
#if 0
        Py_Initialize();
        if (!Py_IsInitialized())
        {
            printf("Py_Initialize error!\n");
            break;
        }
#endif

        PyRun_SimpleString("import sys");
        PyRun_SimpleString("sys.path.append('../pycode')");

        pFile = PyUnicode_FromString("tracker_api");
        pModule = PyImport_Import(pFile);
        if (!pModule)
        {
            printf("PyImport_Import tracker_api.py failed!\n");
            break;
        }

        m_pDict = PyModule_GetDict(pModule);
        if (!m_pDict)
        {
            printf("PyModule_GetDict tracker_api.py failed!\n");
            break;
        }

        pClass = PyDict_GetItemString(m_pDict, "ObjectApi");
        if (!pClass || !PyCallable_Check(pClass))
        {
            printf("PyDict_GetItemString tracker_api failed!\n");
            break;
        }

        m_pHandle = PyObject_CallObject(pClass, nullptr);
        if (!m_pHandle)
        {
            printf("PyInstance_New ObjectApi failed!\n");
            break;
        }
    } while (0);

    if (pClass)
        Py_DECREF(pClass);
    //if (m_pDict)
    //       Py_DECREF(m_pDict);
    if (pModule)
        Py_DECREF(pModule);
    if (pFile)
        Py_DECREF(pFile);

    Py_UNBLOCK_THREADS;
    Py_END_ALLOW_THREADS;
    PyGILState_Release(gstate);

    printf("Engine_api::Engine_api() end!\n");
}


Engine_api::Engine_api(std::string pyClass)
{
    PyObject* pFile = NULL;
    PyObject* pModule = NULL;
    PyObject* pClass = NULL;

    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure(); //申请获取GIL
    Py_BEGIN_ALLOW_THREADS;
    Py_BLOCK_THREADS;

    do
    {
#if 0
        Py_Initialize();
        if (!Py_IsInitialized())
        {
            printf("Py_Initialize error!\n");
            break;
        }
#endif

        PyRun_SimpleString("import sys");
        PyRun_SimpleString("sys.path.append('../pycode')");
        std::cout << pyClass << std::endl;
        pFile = PyUnicode_FromString(pyClass.c_str());
        pModule = PyImport_Import(pFile);
        if (!pModule)
        {
            printf("PyImport_Import tracker_api.py failed!\n");
            break;
        }

        m_pDict = PyModule_GetDict(pModule);
        if (!m_pDict)
        {
            printf("PyModule_GetDict tracker_api.py failed!\n");
            break;
        }

        pClass = PyDict_GetItemString(m_pDict, "ObjectApi");
        if (!pClass || !PyCallable_Check(pClass))
        {
            printf("PyDict_GetItemString tracker_api failed!\n");
            break;
        }

        m_pHandle = PyObject_CallObject(pClass, nullptr);
        if (!m_pHandle)
        {
            printf("PyInstance_New ObjectApi failed!\n");
            break;
        }
    } while (0);

    if (pClass)
        Py_DECREF(pClass);
    //if (m_pDict)
    //       Py_DECREF(m_pDict);
    if (pModule)
        Py_DECREF(pModule);
    if (pFile)
        Py_DECREF(pFile);

    Py_UNBLOCK_THREADS;
    Py_END_ALLOW_THREADS;
    PyGILState_Release(gstate);

    printf("Engine_api::Engine_api() end!\n");
}


Engine_api::~Engine_api()
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure(); //申请获取GIL
    Py_BEGIN_ALLOW_THREADS;
    Py_BLOCK_THREADS;

    if (m_pHandle)
        Py_DECREF(m_pHandle);
    if (m_pDict)
        Py_DECREF(m_pDict);

    Py_UNBLOCK_THREADS;
    Py_END_ALLOW_THREADS;
    PyGILState_Release(gstate);

#if 0
    Py_Finalize();
#endif
    printf("EnginePy::~EnginePy() end!\n");
}

vector<int> Engine_api::get_result(Mat frame, std::string mode)
{
    PyObject *pyResult;

    auto sz = frame.size();
    int x = sz.width;
    int y = sz.height;
    int z = frame.channels();
    uchar *CArrays = new uchar[x*y*z];//这一行申请的内存需要释放指针，否则存在内存泄漏的问题

    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure(); //申请获取GIL
    Py_BEGIN_ALLOW_THREADS;
    Py_BLOCK_THREADS;

    PyObject *ArgList1 = PyTuple_New(1);
    mat2np(frame, ArgList1, CArrays);

    std::string pyMethod = "get_result" + mode;
    pyResult = PyObject_CallMethod(m_pHandle,pyMethod.c_str(),"O",ArgList1);

    Py_DECREF(ArgList1);
    delete []CArrays ;
    CArrays =nullptr;

    Py_UNBLOCK_THREADS;
    Py_END_ALLOW_THREADS;
    PyGILState_Release(gstate);

    vector<int> vret0;
    list2vector(pyResult,vret0);

    return vret0;
}

void Engine_api::test(){
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure(); //申请获取GIL
    Py_BEGIN_ALLOW_THREADS;
    Py_BLOCK_THREADS;

    do
    {
        PyObject_CallMethod(m_pHandle, "test", NULL, NULL);
    } while(0);

    Py_UNBLOCK_THREADS;
    Py_END_ALLOW_THREADS;
    PyGILState_Release(gstate);
}
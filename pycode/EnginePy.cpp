#include "EnginePy.hpp"
#include "utils/mat2numpy.h"
#include <iostream>


Engine_api::Engine_api(){
    PyObject *pFile   = NULL;
    PyObject *pModule = NULL;
    PyObject *pClass  = NULL;

    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure(); //申请获取GIL
    Py_BEGIN_ALLOW_THREADS;
    Py_BLOCK_THREADS;

    do {
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

        pFile   = PyUnicode_FromString("tracker_api");
        pModule = PyImport_Import(pFile);
        if (!pModule) {
            printf("PyImport_Import tracker_api.py failed!\n");
            break;
        }

        m_pDict = PyModule_GetDict(pModule);
        if (!m_pDict) {
            printf("PyModule_GetDict tracker_api.py failed!\n");
            break;
        }

        pClass = PyDict_GetItemString(m_pDict, "ObjectApi");
        if (!pClass || !PyCallable_Check(pClass)) {
            printf("PyDict_GetItemString tracker_api failed!\n");
            break;
        }

        m_pHandle = PyObject_CallObject(pClass, nullptr);
        if (!m_pHandle) {
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

    printf("Engine_api::Engine_api() INIT end!\n");
}


Engine_api::Engine_api(std::string pyClass, int camId){

    PyObject *pFile   = NULL;
    PyObject *pModule = NULL;
    PyObject *pClass  = NULL;
    PyObject *arglist;
    arglist = Py_BuildValue("(i)", camId);

    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure(); //申请获取GIL
    Py_BEGIN_ALLOW_THREADS;
    Py_BLOCK_THREADS;

    do {
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
        pFile   = PyUnicode_FromString(pyClass.c_str());
        pModule = PyImport_Import(pFile);
        if (!pModule) {
            printf("PyImport_Import tracker_api.py failed!\n");
            break;
        }

        m_pDict = PyModule_GetDict(pModule);
        if (!m_pDict) {
            printf("PyModule_GetDict tracker_api.py failed!\n");
            break;
        }

        pClass = PyDict_GetItemString(m_pDict, "ObjectApi");
        if (!pClass || !PyCallable_Check(pClass)) {
            printf("PyDict_GetItemString tracker_api failed!\n");
            break;
        }

        m_pHandle = PyObject_CallObject(pClass, arglist);
        if (!m_pHandle) {
            printf("PyInstance_New ObjectApi failed!\n");
            break;
        }
    } while (0);

    if (pClass)
        Py_DECREF(pClass);
//    if (m_pDict)
//        Py_DECREF(m_pDict);
    if (pModule)
        Py_DECREF(pModule);
    if (pFile)
        Py_DECREF(pFile);
    if (arglist)
        Py_DECREF(arglist);

    Py_UNBLOCK_THREADS;
    Py_END_ALLOW_THREADS;
    PyGILState_Release(gstate);

    printf("Engine_api::Engine_api() INIT end!\n");
}


Engine_api::~Engine_api(){
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure(); //申请获取GIL
    Py_BEGIN_ALLOW_THREADS;
    Py_BLOCK_THREADS;

    if (m_pHandle)
        Py_DECREF(m_pHandle);
//    if (m_pDict)
//        Py_DECREF(m_pDict);

    Py_UNBLOCK_THREADS;
    Py_END_ALLOW_THREADS;
    PyGILState_Release(gstate);

#if 0
    Py_Finalize();
#endif
    printf("EnginePy::~EnginePy() end!\n");
}

void Engine_api::get_result(Mat &frame, std::vector<int> hf_boxs, std::vector<int> trackIDs, std::vector<int> deleteIDs,
                            std::vector<std::vector<int>> ldmk_boxes,
                            std::vector<float> kptsArr, std::vector<float> ageGenderArr){
    PyObject *pyResult;

    auto  sz                  = frame.size();
    int   x                   = sz.width;
    int   y                   = sz.height;
    int   z                   = frame.channels();
    uchar *CArrays            = new uchar[x * y * z];//这一行申请的内存需要释放指针，否则存在内存泄漏的问题
    int   *CArrays_bbox       = new int[hf_boxs.size()];
    int   *CArrays_trackID    = new int[trackIDs.size()];
    int   *CArrays_deleteID   = new int[deleteIDs.size()];
    float *CArrays_kpts       = new float[kptsArr.size()];
    float *CArrays_age        = new float[ageGenderArr.size()];
    int   *CArrays_ldmk_boxes = new int[ldmk_boxes.size() * 6];

    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure(); //申请获取GIL
    Py_BEGIN_ALLOW_THREADS;
    Py_BLOCK_THREADS;

    PyObject *ArgListFrame = PyTuple_New(1);
    mat2np(frame, ArgListFrame, CArrays);

    PyObject *ArgListBBox = PyTuple_New(1);
    vec2np(hf_boxs, ArgListBBox, 6, CArrays_bbox);

    PyObject *ArgListKpts = PyTuple_New(1);
    vec2np(kptsArr, ArgListKpts, 285, CArrays_kpts);

    PyObject *ArgListAge = PyTuple_New(1);
    vec2np(ageGenderArr, ArgListAge, 515, CArrays_age);

    PyObject *ArgListTrackID = PyTuple_New(1);
    vec2np(trackIDs, ArgListTrackID, 1, CArrays_trackID);

    PyObject *ArgListDeleteID = PyTuple_New(1);
    vec2np(deleteIDs, ArgListDeleteID, 1, CArrays_deleteID);

    PyObject *ArgListLdmkBoxes = PyTuple_New(1);
    vec2np(ldmk_boxes, ArgListLdmkBoxes, 6, CArrays_ldmk_boxes);

    std::string pyMethod   = "get_result";
    PyObject    *pRetValue = PyObject_CallMethod(m_pHandle, pyMethod.c_str(), "OOOOOOO", ArgListFrame, ArgListBBox,
                                                 ArgListTrackID,
                                                 ArgListDeleteID, ArgListKpts, ArgListAge, ArgListLdmkBoxes);

    frame = np2mat(pRetValue);
//
    Py_DECREF(ArgListFrame);
    Py_DECREF(ArgListBBox);
    Py_DECREF(ArgListKpts);
    Py_DECREF(ArgListAge);
    Py_DECREF(ArgListTrackID);
    Py_DECREF(ArgListDeleteID);
    Py_DECREF(ArgListLdmkBoxes);
    Py_DECREF(pRetValue);

    delete[]CArrays;
    CArrays = nullptr;

    delete[]CArrays_bbox;
    CArrays_bbox = nullptr;

    delete[]CArrays_trackID;
    CArrays_trackID = nullptr;

    delete[]CArrays_kpts;
    CArrays_kpts = nullptr;

    delete[]CArrays_age;
    CArrays_age = nullptr;

    delete[]CArrays_ldmk_boxes;
    CArrays_ldmk_boxes = nullptr;

    Py_UNBLOCK_THREADS;
    Py_END_ALLOW_THREADS;
    PyGILState_Release(gstate);
}

void Engine_api::test(){
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure(); //申请获取GIL
    Py_BEGIN_ALLOW_THREADS;
    Py_BLOCK_THREADS;

    do {
        PyObject_CallMethod(m_pHandle, "test", NULL, NULL);
    } while (0);

    Py_UNBLOCK_THREADS;
    Py_END_ALLOW_THREADS;
    PyGILState_Release(gstate);
}
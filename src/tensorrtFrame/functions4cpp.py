import os
import sys
import numpy as np
import ctypes
def _permute(vArray, vPerm):
    a = vArray.reshape(-1)
    b = vArray.transpose(vPerm)
    shape = b.shape
    c = b.reshape(-1)
    a[...] = c[...]
    return  shape

def permute(vElementFormat, vArray,  vDim, vPerm):
    dim = np.frombuffer(vDim, dtype=np.int32)
    print("python dim: " , dim)
    array = np.frombuffer(vArray, dtype=vElementFormat).reshape(dim)
    print("src:", array)
    perm = np.frombuffer(vPerm, dtype=np.int32)
    shape = _permute(array, perm)
    print("dst:", array)
    dim[:] = shape[:]
    print("python dim: " , dim)
    return dim.sum()


def getData(vFileName, vPointerSet):
    g_dataSet = np.load(vFileName)
    g_shape = np.zeros(len(g_dataSet.items())+1, dtype=np.int32)
    g_shape[0] = len(g_dataSet.items())
    pointers = np.frombuffer(vPointerSet, dtype=np.int64).reshape(-1)
    pointers[0] = g_shape.ctypes.data
    for i in range(len(g_dataSet.items())):
        arrName = 'arr_' + str(i)
        src = g_dataSet[arrName]
        if False and 1==i:
            src[:, 3:6:2][...] *= 1280
            src[:, 4:7:2][...] *= 720
        print(src.shape)
        pointers[i+1] = src.ctypes.data
        g_shape[i+1] = src.size
    return  float(len(g_dataSet.items()))

def getData0(vFileName, vPointerSet):
    dllPath = '/home/wiwide/VisionProject/framework/tensorrt_framework/cppCommon/build/libCommon.so'
    #print(dllPath, vFileName)
    dataSet = np.load(vFileName)
    #print('load success', len(dataSet.items()))
    dataCoper = ctypes.cdll.LoadLibrary(dllPath)
    pointers = np.frombuffer(vPointerSet, dtype=np.int64).reshape(-1)
    if len(pointers) < len(dataSet.items()) : print('The buffers are too few: '+ str(len(pointers)) + " < " + str(len(dataSet.items())) + " arrays in the file of " + vFileName)
    fsum = 0
    for i in range(len(dataSet.items())):
        arrName = 'arr_' + str(i)
        src = dataSet[arrName] #.reshape(-1).copy()
        src  = src.reshape(-1).copy()
        dst = np.frombuffer(ctypes.c_void_p(pointers[i]), dtype=src.dtype, count=src.count())
        dst[...] = src[...]
        print(dst)
        si = np.abs(src).sum()
        #print(arrName, si, int(pointers[i]))
        #dataCoper.memcpfromPython2cpp(src.ctypes.data_as(ctypes.c_void_p), int(pointers[i]), src.nbytes)
        fsum += float(si)
    return float(fsum)


#a = np.arange(6).reshape(2,3)
#_permute(a, [1,0])
#print (a.reshape(-1))
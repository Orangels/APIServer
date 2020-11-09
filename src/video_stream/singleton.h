#ifndef SINGLETON_LS_H
#define SINGLETON_LS_H
#include <iostream>
#include <mutex>
using namespace std;

template <typename T>
class Singleton {
public:
    static T * GetInstance() {
        if (pSingle == nullptr) {
            lock_guard<mutex> lock(mtx);
            if (pSingle == nullptr) {
                pSingle = new T();
            }
        }
        return pSingle;
    }

    static T * GetInstance(const std::string& params) {
        if (pSingle == nullptr) {
            lock_guard<mutex> lock(mtx);
            if (pSingle == nullptr) {
                pSingle = new T(params);
            }
        }
        return pSingle;
    }
private:
    Singleton() {}
    virtual ~Singleton()  {}

    static T * pSingle;
    static mutex mtx;
};

template <typename T>
mutex Singleton<T>::mtx;
template <typename T>
T* Singleton<T>::pSingle = nullptr;

#endif
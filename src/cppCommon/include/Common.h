#pragma once
#include <string>
#include <sstream>
#include <vector>
#include <ostream>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <ctime>
#include <ratio>
#include <chrono>
#include "Singleton.h"
#include "StringFunction.h"
/*
#if (define WIN32) || (define _WIN32)
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#define NEW_WITH_MEMORY_LEAK_CHECKING new(_NORMAL_BLOCK,__FILE__,__LINE__)
#define new    NEW_WITH_MEMORY_LEAK_CHECKING
//_CrtDumpMemoryLeaks(); at the end of main function
#ifdef _Common_Dll_Exports
#define _Common_Dll_Export __declspec(dllexport)
#else
#define _Common_Dll_Export __declspec(dllimport)
#endif
#else
#define _Common_Dll_Export
#endif
*/
#define _Common_Dll_Export
unsigned int getThreadID();


struct _Common_Dll_Export SSystemTime 
{
	void refresh();
	SSystemTime(){ refresh(); }
	unsigned short wYear,wMonth,wDayOfWeek,wDay;
	unsigned short wHour,wMinute,wSecond,wMilliseconds;
	const std::string outputDate(char vSplit = '_');
	const std::string outputTime(char vSplit=':');
	const std::string outputDateHour(char vSplit = '_');
	friend std::ostream& operator <<(std::ostream& Os, const SSystemTime& vST)
	{
		Os << vST.wYear << "_" << vST.wMonth << "_" << vST.wDay << " : " << vST.wHour << "_" << vST.wMinute << "_" << vST.wSecond << "_" << vST.wMilliseconds;
		return Os;
	}
};

class CTimer
{
public:
	CTimer():m_TimeSpan(0),m_count(0){ }
	void start(int vCount = 1) { m_Begin = std::chrono::steady_clock::now(); m_count += vCount; }
	void stop(){ m_End = std::chrono::steady_clock::now(); m_TimeSpan += std::chrono::duration_cast<std::chrono::duration<float>>(m_End-m_Begin).count(); }
	float getAvgMillisecond() { return m_TimeSpan * 1000 / m_count; }
	float getTimeSpanSecond() { return m_TimeSpan; }
	size_t getCount() { return m_count; }
	static void sleep(unsigned int vMiliSeconds){std::this_thread::sleep_for(std::chrono::milliseconds(vMiliSeconds));}
	void reset() { m_TimeSpan = 0; m_count = 0; }
private:
	std::chrono::steady_clock::time_point m_Begin, m_End;
	float m_TimeSpan=0;
	size_t m_count = 0;
};


class _Common_Dll_Export CConfiger : protected CSingleton<CConfiger>
{
public: 
	virtual ~CConfiger(){m_ConfigDataMap.clear();}
	const std::string& readValue(const std::string& vKey);
	template<typename T>
	void readValue(const std::string& vKey, T&voValue)
	{
		std::stringstream m_Convertor(readValue(vKey));
		m_Convertor >> voValue;
	}

	template<typename T>
	void readValue(const std::string& vKey, std::vector<T>&voValues, char vSeparator = ' ', bool vClearBeforAppanding = true)
	{
		std::string values = readValue(vKey);
		splitStringAndConvert(values, voValues, vSeparator, ' ', vClearBeforAppanding);
	}

	template<typename T>
	T readValue(const std::string& vKey)
	{
		T voValue;
		std::stringstream m_Convertor(readValue(vKey));
		m_Convertor >> voValue;
		return voValue;
	}

	void appandConfigInfor(const std::string& vConfigFileName);
	static CConfiger* getOrCreateConfiger(const std::string& vConfigFileName = "");//configer.ini

private:
	CConfiger(){};
	friend class CSingleton<CConfiger>;
	void __parseConfigInfor(std::ifstream& vIFS);
	std::unordered_map<std::string, std::string> m_ConfigDataMap;
};

class _Common_Dll_Export CLog : public CSingleton<CLog>
{
public:
	CLog();
	~CLog();

template<class T> void output(const T &v)
{
	if (m_multiThreadMutex) m_mutex.lock();
	__outputCurrentTime();
	if (m_TxtRequire) m_TxtAppander << v << std::endl;
	if (m_StandRequire) std::cout << v << std::endl;
	if (m_multiThreadMutex) m_mutex.unlock();
}
template<class T1, class T2> void output(const T1& v1, const T2 &v2)
{
	if (m_multiThreadMutex) m_mutex.lock();
	__outputCurrentTime();
	if (m_TxtRequire) m_TxtAppander << v1 << v2 << std::endl;
	if (m_StandRequire) std::cout << v1 << v2 << std::endl;
	if (m_multiThreadMutex) m_mutex.unlock();
}
template<class T1, class T2, class T3> void output(const T1& v1, const T2 &v2, const T3 &v3)
{
	if (m_multiThreadMutex) m_mutex.lock();
	__outputCurrentTime();
	if (m_TxtRequire) m_TxtAppander << v1 << v2 << v3 << std::endl;
	if (m_StandRequire) std::cout << v1 << v2 << v3 << std::endl;
	if (m_multiThreadMutex) m_mutex.unlock();
}

template<class T1, class T2, class T3, class T4> void output(const T1& v1, const T2 &v2, const T3 &v3, const T4 &v4)
{
	if (m_multiThreadMutex) m_mutex.lock();
	__outputCurrentTime();
	if (m_TxtRequire) m_TxtAppander << v1 << v2 << v3 << v4 << std::endl;
	if (m_StandRequire) std::cout << v1 << v2 << v3 << v4 << std::endl;
	if (m_multiThreadMutex) m_mutex.unlock();
}
template<class T1, class T2, class T3, class T4, class T5> void output(const T1& v1, const T2 &v2, const T3 &v3, const T4 &v4, const T5 &v5)
{
	__outputCurrentTime();
	if (m_TxtRequire) m_TxtAppander << v1 << v2 << v3 << v4 << v5 << std::endl;
	if (m_StandRequire) std::cout << v1 << v2 << v3 << v4 << v5 << std::endl;
}
template<class T1, class T2, class T3, class T4, class T5, class T6> void output(const T1& v1, const T2 &v2, const T3 &v3, const T4 &v4, const T5 &v5, const T6 &v6)
{
	if (m_multiThreadMutex) m_mutex.lock();
	__outputCurrentTime();
	if (m_TxtRequire) m_TxtAppander << v1 << v2 << v3 << v4 << v5 << v6 << std::endl;
	if (m_StandRequire) std::cout << v1 << v2 << v3 << v4 << v5 << v6 << std::endl;
	if (m_multiThreadMutex) m_mutex.unlock();
}

template<class T1, class T2, class T3, class T4, class T5, class T6, class T7> void output(const T1& v1, const T2 &v2, const T3 &v3, const T4 &v4, const T5 &v5, const T6 &v6, const T7 &v7)
{
	__outputCurrentTime();
	if (m_TxtRequire) m_TxtAppander << v1 << v2 << v3 << v4 << v5 << v6 << v7 << std::endl;
	if (m_StandRequire) std::cout << v1 << v2 << v3 << v4 << v5 << v6 << v7 << std::endl;
}

template<class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8> void output(const T1& v1, const T2 &v2, const T3 &v3, const T4 &v4, const T5 &v5, const T6 &v6, const T7 &v7, const T8 &v8)
{
	if (m_multiThreadMutex) m_mutex.lock();
	__outputCurrentTime();
	if (m_TxtRequire) m_TxtAppander << v1 << v2 << v3 << v4 << v5 << v6 << v7 << v8 << std::endl;
	if (m_StandRequire) std::cout << v1 << v2 << v3 << v4 << v5 << v6 << v7 << v8 << std::endl;
	if (m_multiThreadMutex) m_mutex.unlock();
}

	void flush();
	void setOutPutOption(bool vTxt, bool vStand);
	void setLogFileName(const std::string& vFileName);
	void setLogTime(bool vRequire = true){ m_TimeRequire = vRequire; }
	void setOutputMutex(bool vMultThreadMutex) { m_multiThreadMutex = vMultThreadMutex; }

private:

	std::ofstream m_TxtAppander;
	std::mutex    m_mutex;
	bool m_TxtRequire, m_StandRequire, m_TimeRequire, m_multiThreadMutex;
	SSystemTime m_SystemTime;
	void __outputCurrentTime();
	std::string m_logFileName;
	friend class CSingleton <CLog> ;
};


class CBaseFactory;
class _Common_Dll_Export CFactoryDirectory : public CSingleton<CFactoryDirectory>
{
public:
	virtual ~CFactoryDirectory(void);
	void* createProduct(std::string vSig, int velmentCount=1); 
	int getSizeof(std::string vSig);
	void registerFactory(CBaseFactory *vpFactory, const std::string& vSig);
	void setDllSearchPath(const std::string& vDllPath = "./");
	bool existFactory(std::string& vSigName);

protected:
	CFactoryDirectory(void);

private:
	std::unordered_map<std::string, CBaseFactory*>::iterator __findFactory(std::string& vSigName);
	CLog* m_pLog;
	std::string m_DllPath;
	std::vector<void*> m_DllSet;  
	std::unordered_map<std::string, CBaseFactory*> m_FactoryMap;

	friend class CSingleton<CFactoryDirectory>;
};

class CBaseFactory
{
public:
	CBaseFactory(){}
	virtual ~CBaseFactory(){}

protected:
	virtual void* _createProductV(int velmentCount) {return NULL;}
	virtual int _sizeof() { return 0; }
	friend class CFactoryDirectory;
};

template <class TDerivedClass> class CFactory : public CBaseFactory
{
public:
	CFactory(const std::string& vSig)
	{
		CFactoryDirectory::getOrCreateInstance()->registerFactory(this, vSig);
	}
	~CFactory(void) {}

protected:
	virtual void* _createProductV(int velmentCount) override {return new TDerivedClass[velmentCount]; }
	virtual int _sizeof() { return sizeof(TDerivedClass); }
};

class CRandom
{
public:
	CRandom(bool vUseTimeSeed = true) :m_useTimeSeed(vUseTimeSeed)
	{
		if (vUseTimeSeed)
			srand((unsigned)time(NULL));
	}
	~CRandom();
	template<typename RealType>
	RealType getRandomNumber(RealType vBegin, RealType vEnd)
	{
		RealType Lenght = vEnd - vBegin;
		RealType Value = __random() * Lenght + vBegin;
		return Value;
	}
	template<typename RealType>
	RealType getRandomNumber(RealType vEnd)
	{
		return getRandomNumber((RealType)0, vEnd);
	}

private:
	inline double __random() {
		int Rand = 0;
		while (RAND_MAX == Rand)Rand = rand();
		return Rand / double(RAND_MAX);
	}
	bool m_useTimeSeed;
};

#include <list>
#include<condition_variable>
class CSemaphore
{
public:
	CSemaphore(int vmaxResourceNum, int vNumResource = 1) : m_numResource( vNumResource), m_wakeups( 0), m_offset(0), m_maxResourceNum(vmaxResourceNum) {}
	
	int wait(int vNumResource = 1);
	void signal(int vNumResource = 1);

private:
	int m_offset, m_maxResourceNum;
	int m_wakeups;
	int m_numResource;
	std::mutex m_mutex;
	std::list<int> m_deficient;
	std::condition_variable m_condition;
};

template<typename T>
double getSum(T* p, int n, bool v4debug=true, bool vAbs = true)
{
    double sumN = 0;
	T* pend = p + n;
	int step = 1;
    while (p<pend)
    {
        T temp = *p;
        if (vAbs && temp < 0) if (v4debug)temp = 0;else temp *= -1;
        sumN += temp;
		if (v4debug) p += step++;
		else p++;
		if (step > 97) step = 1;
    }
    return sumN;
}

template<typename T>
void outPutSum(T* p, int n, const std::string& vInfor)
{
	double sumBug = getSum(p, n);
	double sumAll = getSum(p, n, false);
	std::cout << vInfor << " : " << sumAll << ", " << sumBug << std::endl;
}
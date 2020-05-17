#pragma once
#include <string>
#include <iostream>
#include "FileFunction.h"
#include "Common.h"


void CLog::setOutPutOption(bool vTxt, bool vStand)
{
	m_TxtRequire = vTxt;
	m_StandRequire = vStand;
}

CLog::CLog() : m_TxtRequire(true), m_StandRequire(true), m_TimeRequire(true),m_multiThreadMutex(false)
{
	if (!isExist("./logFiles/"))
		system("mkdir ./logFiles/");
	m_logFileName = "./logFiles/" + m_SystemTime.outputDateHour()+"_"+std::to_string(getThreadID()) + ".txt";
}

void CLog::flush()
{
	m_TxtAppander.flush();
}

CLog::~CLog()
{
	m_TxtAppander.close();
}

void CLog::__outputCurrentTime()
{
	if (m_TimeRequire)
	{
		auto Time = m_SystemTime.outputTime();
		if (m_TxtRequire) m_TxtAppander << Time << std::endl;
		if (m_StandRequire) std::cout << Time << std::endl;;
	}
	if (m_TxtRequire && !m_TxtAppander.is_open())
	{
		setLogFileName(m_logFileName);
		if (m_TxtRequire && !m_TxtAppander.is_open())
			m_TxtRequire = false;
	}
}

void CLog::setLogFileName(const std::string& vFileName)
{
	if (vFileName.empty())
		return;
	m_logFileName = vFileName;
	if (m_TxtRequire)
	{
		if (m_TxtAppander.is_open())
			m_TxtAppander.close();
		openFileStream(m_logFileName, std::ios::app, m_TxtAppander, "Fail to  open log file of " + m_logFileName);
	}
}

const std::string& CConfiger::readValue(const std::string& vKey)
{
	auto iValue = m_ConfigDataMap.find(toUpperCopy(vKey));
	if (iValue != m_ConfigDataMap.end())
		return iValue->second;
	static std::string NoValue = "";
	return NoValue;
}

void CConfiger::__parseConfigInfor(std::ifstream& vIFS)
{
	std::string Line;
	std::vector < std::string > KeyValue;
	while (getline(vIFS, Line))
	{
		trim(Line);
		if (!Line.empty() && false == isStartWith(Line, "//"))
		{
			splitString(Line, KeyValue, ' ');
			if (KeyValue.size() > 1)
				m_ConfigDataMap[toUpperCopy(KeyValue.front())] = KeyValue[1];
		}
	}
}

void CConfiger::appandConfigInfor(const std::string& vConfigFileName)
{
	std::string ConfigFileName = toUpperCopy(vConfigFileName);
	std::string& Flag = m_ConfigDataMap[ConfigFileName];
	if (Flag != "") Flag = "Flag";
	else
	{
		std::ifstream IFS(vConfigFileName);
		if (IFS.fail())
		{
			CLog *pLog = CLog::getOrCreateInstance();
			pLog->setOutPutOption(true, true);
			pLog->setLogTime(true);
			pLog->output("Can't open configure file: " + vConfigFileName);
		}
		else
		{
			__parseConfigInfor(IFS);
		}
		IFS.close();
	}
}

CConfiger* CConfiger::getOrCreateConfiger(const std::string& vConfigFileName /*= "Configer.txt"*/)
{
	CConfiger* pConfig = getOrCreateInstance();
	if (isExist(vConfigFileName))
		pConfig->appandConfigInfor(vConfigFileName);
	return pConfig;
}

void SSystemTime::refresh()
{
	auto tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	struct tm* ptm = localtime(&tt);
	this->wYear = ptm->tm_year+1900;
	this->wMonth = ptm->tm_mon+1;
	this->wDay = ptm->tm_mday;
	this->wHour = ptm->tm_hour;
	this->wMinute = ptm->tm_min;
	this->wSecond = ptm->tm_sec;
	std::chrono::milliseconds now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
	this->wMilliseconds = now_ms.count() % 1000;
}

const std::string SSystemTime::outputDate(char vSplit/*='_'*/)
{
	refresh();
	std::string Date;
	Date.resize(10);
	sprintf((char*)Date.c_str(), "%4d%c%2d%c%2d", wYear, vSplit, wMonth, vSplit, wDay);
	return Date;
}

const std::string SSystemTime::outputTime(char vSplit/*=':'*/)
{
	refresh();
	std::string Time;
	Time.resize(12);
	sprintf((char*)Time.c_str(), "%2d%c%2d%c%2d%c%3d", wHour, vSplit, wMinute, vSplit, wSecond, vSplit, wMilliseconds);
	return Time;
}


const std::string SSystemTime::outputDateHour(char vSplit /*= '_'*/)
{
	refresh();
	std::string DateHour;
	DateHour.resize(13);
	int Lenth = sprintf((char*)DateHour.c_str(), "%d%c%d%c%d%c%d", wYear, vSplit, wMonth, vSplit, wDay, vSplit, wHour);
	DateHour.resize(Lenth);
	return DateHour;
}

CFactoryDirectory::CFactoryDirectory(void) : m_pLog(CLog::getOrCreateInstance())
{

}

std::unordered_map<std::string, CBaseFactory*>::iterator CFactoryDirectory::__findFactory(std::string& vSigName)
{
	trim(vSigName);
	toUpper(vSigName);
	return m_FactoryMap.find(vSigName);
}

int CFactoryDirectory::getSizeof(std::string vSig)
{
	auto iFactory = __findFactory(vSig);
	if (iFactory == m_FactoryMap.end())
		return 0;
	return iFactory->second->_sizeof();
}

void* CFactoryDirectory::createProduct(std::string vSig, int velmentCount)
{
	auto iFactory = __findFactory(vSig);
	if (iFactory  == m_FactoryMap.end())
		return NULL;
	if (m_FactoryMap.end() == iFactory)
	{
#ifdef _DEBUG
		vSig += "D.dll";
#else
#ifdef DEBUG
		vSig += "D.dll";
#endif // DEBUG
		vSig += ".dll";
#endif // DEBUG
		vSig = assembleFullFileName(m_DllPath, vSig);
// 		HINSTANCE hInstLibrary = ::LoadLibrary(vSig.c_str());
// 		if (NULL != hInstLibrary)
// 		{
// 			m_DllSet.push_back(hInstLibrary);
// 			iFactory = m_FactoryMap.find(vSig);
// 		}
// 		else
// 		{
// 			m_pLog->output("Can not find dll file : " + vSig);
// 		}
	}

	if (iFactory != m_FactoryMap.end())
		return iFactory->second->_createProductV(velmentCount);
	return  NULL;
}

void CFactoryDirectory::registerFactory(CBaseFactory *vpFactory, const std::string& vSig)
{
	m_FactoryMap[toUpperCopy(vSig)] = vpFactory;
}

void CFactoryDirectory::setDllSearchPath(const std::string& vDllPath /*= "./"*/)
{
	m_DllPath = vDllPath;
}

bool CFactoryDirectory::existFactory(std::string& vSigName)
{
	auto i = __findFactory(vSigName);
	return i!=m_FactoryMap.end();
}

CFactoryDirectory::~CFactoryDirectory(void)
{
	while (m_DllSet.size())
	{
// 		FreeLibrary(m_DllSet.back());
		m_DllSet.pop_back();
	}
}

unsigned int getThreadID()
{
	auto Tid = std::this_thread::get_id();
	return *(unsigned int*)&Tid;
}

int CSemaphore::wait(int vNumResource)
{
	if (vNumResource < 1) return m_offset << 8;
	if (vNumResource > 255) std::cout << "ask for too much resource." << std::endl;
	int offset = 0;
	{
		std::unique_lock<std::mutex>lock{ m_mutex };
		offset = m_offset; 
		if (m_numResource < 1)
		{
retry:		m_deficient.push_back(vNumResource);
			m_condition.wait(lock, [&]()->bool {return m_wakeups > 0; });
			--m_wakeups;// ok, a wakeup was consumed!
			if (m_deficient.size() && m_deficient.front() < 0)
			{
				vNumResource += m_deficient.front();
				if (vNumResource < 1) vNumResource = 1;
				m_deficient.pop_front();
			}
			if (m_numResource < 1) goto retry;
		}
		if (m_numResource < vNumResource) 
			vNumResource = m_numResource;
		m_numResource -= vNumResource;
		m_offset = (m_offset + vNumResource) % m_maxResourceNum;
	}
	offset <<= 8;
	offset += vNumResource;

	return offset;
}

void CSemaphore::signal(int vNumResource)
{
	std::lock_guard<std::mutex>lock{ m_mutex };
	if (vNumResource < 1) return;
	m_numResource += vNumResource;
	while (m_deficient.size() && vNumResource>0)
	{
		if (m_deficient.front() < 0) m_deficient.front() *= -1;
		vNumResource -= m_deficient.front();
		if (vNumResource >= 0)
			m_deficient.pop_front();
		else 
			m_deficient.front() = vNumResource;
		++m_wakeups;
		m_condition.notify_one();
	}
}



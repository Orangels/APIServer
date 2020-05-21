#include <iostream>
#include "dispatch.h"
#include "Common.h"
using namespace std;

int main(void)
{
	cout<<("Dispatch Server start\n")<<endl;
    CConfiger* pConfiger = CConfiger::getOrCreateConfiger("../cfg/configer.txt");
    Dispatch dispatch;
//    dispatch.test();
//    dispatch.engine_Test();
    dispatch.multithreadTest();
	return 0;
}
#include <iostream>
#include "dispatch.h"
using namespace std;

int main(void)
{
	cout<<("Dispatch Server start\n")<<endl;
    Dispatch dispatch;
//    dispatch.test();
//    dispatch.engine_Test();
    dispatch.multithreadTest();
	return 0;
}
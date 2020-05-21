#include "FileFunction.h"
#include "postPara3DKeyPoints.pb.h"
#include "protoIO.h"

int main(int argc, char** argv)
{
	std::string binaryProtoBufferFileName;
	inputeParameter(binaryProtoBufferFileName, argc, argv, "Please input binary protoBuffer file name.");
	CProtoIO pio;
	C3DPara para;
	pio.readProtoFromBinaryFile(binaryProtoBufferFileName, &para);
	std::string name = deleteExtentionName(binaryProtoBufferFileName, true);
	name += "txt";
	pio.writeProtoToTextFile(para, name);
	return 0;
}
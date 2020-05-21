import os
import sys
import numpy as np
import pdb
from argparse import ArgumentParser

if __name__ == '__main__':
    #pdb.set_trace()
    parser = ArgumentParser(description="Create the tool that convert binary protobuffer to text.")
    parser.add_argument('--pro', default="./example.proto", help="The proto file name including its path.")
    parser.add_argument('--dat', default="C3DPara", help="The outmost data structure name in proto file name.")
    args = parser.parse_args()
    path = os.path.dirname(args.pro)
    if path == "" or path is None or len(path)<1:
        path = '.'
    fileName = os.path.basename(args.pro)
    protoName =os.path.splitext(fileName)[0]
    exeFileName =  protoName + "ProtobufferConvertor"

    cmd = "protoc  -I="+ path + " " +  args.pro
    os.system(cmd+" --cpp_out="+path)
    os.system(cmd+" --python_out="+path)

    #protobufferConvertor
    #  modify main.cpp CMakeLists.txt
    os.system("cp main.cpp0 main.cpp")
    os.system("cp CMakeLists.txt0 CMakeLists.txt")
    os.system("sed -i 's/" + "example" + "/"+protoName +"/g' " + "CMakeLists.txt")
    os.system("sed -i 's/" + "protobufferConvertor" + "/"+exeFileName +"/g' " + "CMakeLists.txt")
    os.system("sed -i 's/" + "example" + "/"+protoName +"/g' " + "main.cpp")
    os.system("sed -i 's/" + "C3DPara" + "/"+args.dat +"/g' " + "main.cpp")



    os.system("sh ./createExe.sh " + exeFileName)





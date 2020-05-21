#!/bin/bash
cd $1 
python3 create.py --pro postPara3DKeyPoints.proto --dat C3DPara
cp postPara3DKeyPoints.pb.* ..
cp postPara3DKeyPoints_pb2.py ../../../..
sed -i 's/execute_process(COMMAND sh/#execute_process(COMMAND sh/g' CMakeLists.txt


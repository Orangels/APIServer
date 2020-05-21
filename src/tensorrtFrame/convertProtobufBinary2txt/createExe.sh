#!/bin/bash
mkdir build
cd build
rm * -fr
cmake ..
echo "make..."
make -j2
cp $1 ..

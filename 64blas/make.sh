#!/bin/sh
BLASPATH="/home/wsk/openblas_lib"
rm test
g++ -g3 -fpermissive -w test.cpp exec.cpp -static -I$BLASPATH/include/ $BLASPATH/lib/libopenblas.a -lpthread -o  test

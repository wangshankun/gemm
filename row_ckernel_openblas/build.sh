#!/bin/sh
rm -rf *.o *.a test

g++ -g3 -w test.cpp MMult_4x4_13.cpp -I/home/wsk/openblas_lib/include /home/wsk/openblas_lib/lib/libopenblas.a  -lpthread -o  test

./test

rm -rf *.o  *.a

#!/bin/sh -xe

CXX="${CXX:-clang++}"

CFLAGS="-O3 -std=c++11 -march=native -mtune=native -Wall -Wextra -pedantic"

SRCS=$(ls examples/*.cpp)

mkdir -p build

for SRC in $SRCS
do
  OUTPUT="build/$(basename ${SRC%.*})"
  $CXX $CFLAGS -o $OUTPUT $SRC
done

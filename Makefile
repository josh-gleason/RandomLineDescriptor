FLAGS=`pkg-config opencv --cflags` -std=c++11
LIBS=`pkg-config opencv --libs`

all: bin/main bin/test

bin/main: src/main.cc Makefile
	g++ ${FLAGS} src/main.cc ${LIBS} -o bin/main

bin/test: src/mser/test.cpp Makefile
	g++ src/mser/test.cpp src/mser/mser.cpp -ljpeg -o bin/test


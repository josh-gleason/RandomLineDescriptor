FLAGS=`pkg-config opencv --cflags` -std=c++11 -g
LIBS=`pkg-config opencv --libs` -lboost_program_options -lboost_system -lboost_filesystem

all: bin/main

bin/main: src/main.cc src/settings.h src/results.h build/settings.o Makefile
	g++ ${FLAGS} src/main.cc build/settings.o ${LIBS} -o bin/main

build/settings.o: src/settings.h src/results.h src/settings.cc Makefile
	g++ -c src/settings.cc -o build/settings.o ${FLAGS}

bin/test: src/mser/test.cpp Makefile
	g++ src/mser/test.cpp src/mser/mser.cpp -ljpeg -o bin/test


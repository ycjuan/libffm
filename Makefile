CXX = g++
CXXFLAGS = -Wall -O3 -std=c++0x -march=native

# comment the following flags if you do not want to use OpenMP
DFLAG += -DUSEOMP
CXXFLAGS += -fopenmp

all: ffm-train ffm-predict

ffm-train: ffm-train.cpp ffm.o
	$(CXX) $(CXXFLAGS) -o $@ $^

ffm-predict: ffm-predict.cpp ffm.o
	$(CXX) $(CXXFLAGS) -o $@ $^

ffm.o: ffm.cpp ffm.h
	$(CXX) $(CXXFLAGS) $(DFLAG) -c -o $@ $<

clean:
	rm -f ffm-train ffm-predict ffm.o

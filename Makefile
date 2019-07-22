CXX = g++
CXXFLAGS = -Wall -O3 -std=c++0x -march=native

ifneq ($(USESSE), OFF)
	DFLAG += -DUSESSE
endif

ifneq ($(USEOMP), OFF)
	DFLAG += -DUSEOMP
	OMP_CXXFLAGS ?= -fopenmp
	CXXFLAGS += $(OMP_CXXFLAGS)
endif

all: ffm-train ffm-predict

ffm-train: ffm-train.cpp ffm.o timer.o
	$(CXX) $(CXXFLAGS) $(DFLAG) $(OMP_LDFLAGS) -o $@ $^

ffm-predict: ffm-predict.cpp ffm.o timer.o
	$(CXX) $(CXXFLAGS) $(DFLAG) $(OMP_LDFLAGS) -o $@ $^

ffm.o: ffm.cpp ffm.h timer.o
	$(CXX) $(CXXFLAGS) $(DFLAG) -c -o $@ $<

timer.o: timer.cpp timer.h
	$(CXX) $(CXXFLAGS) $(DFLAG) -c -o $@ $<

clean:
	rm -f ffm-train ffm-predict ffm.o timer.o

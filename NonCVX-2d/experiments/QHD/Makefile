all: 2d-pseudospec

CXX := g++
CXXFLAGS := -O3 --std=c++17

2d-pseudospec: 2d-pseudospec.o to_npy.o
	$(CXX) $(CXXFLAGS) -o 2d-pseudospec $^ -fopenmp -lfftw3_omp -lfftw3 -lm -lmatio -lstdc++fs

2d-pseudospec.o: 2d-pseudospec.cpp
	$(CXX) $(CXXFLAGS) -c $^ -fopenmp

to_npy.o: to_npy.hpp to_npy.cpp
	$(CXX) $(CXXFLAGS) -c $^

clean:
	rm -f 2d-pseudospec 2d-pseudospec.o to_npy to_npy.o to_npy.hpp.gch
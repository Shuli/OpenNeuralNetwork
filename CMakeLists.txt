
# @author Hisashi Ikari<ikari@akane.waseda.jp>, Waseda University.

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT("Open Neural Network" CXX)

SET(PJ_VERSION "1.0.0")
SET(CMAKE_BUILD_TYPE Release)

SET(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3")
SET(CMAKE_CXX_FLAGS_DEBUG "-Wall -g")

SET(SO_SOURCES
Neuron.cpp
Synaps.cpp
Prim.cpp
Network.cpp
)

SET(SO_HEADERS
Synaps.h
Neuron.h
Prim.h
Network.h
)

SET(BI_SOURCES
Reasoning.cpp
)

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})
LINK_DIRECTORIES(${PROJECT_LIBRARY_DIR})

ADD_LIBRARY(NeuralBase SHARED ${SO_SOURCES})
ADD_EXECUTABLE(NeuralNet ${BI_SOURCES})
TARGET_LINK_LIBRARIES(NeuralNet NeuralBase)



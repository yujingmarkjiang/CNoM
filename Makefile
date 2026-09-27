# SPDX-License-Identifier: Apache-2.0
CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -I runtime/include
BUILD ?= /tmp/cnom-build
export PATH := /usr/local/go/bin:$(PATH)

.PHONY: all runtime test cloud clean

all: runtime cloud

runtime:
	mkdir -p $(BUILD)/runtime
	$(CXX) $(CXXFLAGS) -c runtime/src/gate.cpp -o $(BUILD)/runtime/gate.o
	$(CXX) $(CXXFLAGS) -c runtime/src/router.cpp -o $(BUILD)/runtime/router.o
	$(CXX) $(CXXFLAGS) runtime/tests/test_runtime.cpp $(BUILD)/runtime/gate.o $(BUILD)/runtime/router.o -o $(BUILD)/runtime/cnom-runtime-test
	$(CXX) $(CXXFLAGS) runtime/src/edge_main.cpp $(BUILD)/runtime/gate.o $(BUILD)/runtime/router.o -o $(BUILD)/runtime/cnom-edge
	$(BUILD)/runtime/cnom-runtime-test

cloud:
	go test ./...
	go build -o $(BUILD)/cnomd ./cmd/cnomd
	go build -o $(BUILD)/cnom-exp ./cmd/cnom-exp

test: runtime cloud

clean:
	rm -rf $(BUILD)

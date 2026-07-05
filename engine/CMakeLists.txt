cmake_minimum_required(VERSION 3.15)
project(AgnosMLEngine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Protobuf and gRPC packages
find_package(gRPC CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)

# Define directories
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(PROTO_GEN_DIR ${SRC_DIR}/proto_gen)

# Include directories so the compiler can find our headers
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROTO_GEN_DIR}
)

# Define the source files, including our generated Protobuf/gRPC stubs
set(SOURCES
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/backend/cpu_backend.cpp
    ${SRC_DIR}/primitives/tensor.cpp
    ${SRC_DIR}/core/model_loader.cpp
    ${SRC_DIR}/core/neural_network.cpp
    ${PROTO_GEN_DIR}/inference.pb.cc
    ${PROTO_GEN_DIR}/inference.grpc.pb.cc
)

# Create the executable
add_executable(agnos_engine ${SOURCES})

# Link the required libraries (gRPC and Protobuf)
target_link_libraries(agnos_engine
    gRPC::grpc++
    gRPC::grpc++_reflection
    protobuf::libprotobuf
)
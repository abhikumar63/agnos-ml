#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>
#include "inference.grpc.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using agnos::inference::InferenceEngine;
using agnos::inference::HealthRequest;
using agnos::inference::HealthResponse;

// Implement the gRPC Service
class InferenceEngineServiceImpl final : public InferenceEngine::Service {
    Status HealthCheck(ServerContext* context, const HealthRequest* request, HealthResponse* reply) override {
        reply->set_status(HealthResponse::SERVING);
        reply->set_active_models_count(0); // We haven't loaded the .agnos file yet
        return Status::OK;
    }
    
    // We will implement Predict, BatchPredict, and LoadModel in Milestone 2
};

void RunServer() {
    std::string server_address("0.0.0.0:50051");
    InferenceEngineServiceImpl service;

    ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    // Register "service" as the instance through which we'll communicate with clients.
    builder.RegisterService(&service);
    
    // Finally assemble the server.
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "🚀 Agnos-ML C++ Compute Engine listening on " << server_address << std::endl;

    // Wait for the server to shutdown.
    server->Wait();
}

int main(int argc, char** argv) {
    RunServer();
    return 0;
}
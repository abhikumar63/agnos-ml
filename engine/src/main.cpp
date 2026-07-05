#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>
#include "inference.grpc.pb.h"

#include "core/model_loader.hpp"
#include "backend/cpu_backend.hpp"
#include "core/neural_network.hpp"
#include "primitives/tensor.hpp"

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

void TestInference() {
    std::cout << "\n--- Running Local Inference Test ---" << std::endl;
    try {
        // 1. Load the model from disk (adjust path if necessary)
        std::string model_path = "../../data/models/v1_base_mlp.agnos";
        auto layers = agnos::core::ModelLoader::load_agnos(model_path);
        
        // 2. Initialize the backend and network
        auto backend = std::make_shared<agnos::backend::CPUBackend>();
        agnos::core::NeuralNetwork nn(std::move(layers), backend);

        // 3. Create dummy input data (Batch size: 1, Features: 20)
        // Filling it with 1.0f just to see deterministic math output
        agnos::primitives::Tensor input({1, 20});
        for(size_t i = 0; i < 20; ++i) {
            input.at(0, i) = 1.0f; 
        }

        std::cout << "Pushing Tensor of shape [1, 20] through network..." << std::endl;

        // 4. Run the forward pass!
        auto output = nn.forward(input);

        // 5. Output the results
        std::cout << "Prediction Output Shape: [" << output.shape()[0] << ", " << output.shape()[1] << "]" << std::endl;
        std::cout << "Prediction Values: [";
        for(size_t i = 0; i < output.size(); ++i) {
            std::cout << output[i] << (i < output.size() - 1 ? ", " : "");
        }
        std::cout << "]\n------------------------------------\n" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Inference Test Failed: " << e.what() << std::endl;
    }
}

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
    TestInference();
    RunServer();
    return 0;
}
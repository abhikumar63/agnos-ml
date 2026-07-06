#include <iostream>
#include <memory>
#include <string>
#include <chrono>

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
using agnos::inference::PredictRequest;
using agnos::inference::PredictResponse;

// Implement the gRPC Service
class InferenceEngineServiceImpl final : public InferenceEngine::Service {
private:
    std::shared_ptr<agnos::core::NeuralNetwork> network_;

public:
    // Inject the loaded network into the gRPC service
    InferenceEngineServiceImpl(std::shared_ptr<agnos::core::NeuralNetwork> network) 
        : network_(std::move(network)) {}

    Status HealthCheck(ServerContext* context, const HealthRequest* request, HealthResponse* reply) override {
        reply->set_status(HealthResponse::SERVING);
        reply->set_active_models_count(network_ ? 1 : 0); 
        return Status::OK;
    }
    
    Status Predict(ServerContext* context, const PredictRequest* request, PredictResponse* reply) override {
        if (!network_) {
            return grpc::Status(grpc::StatusCode::UNAVAILABLE, "Neural network is not loaded.");
        }

        try {
            // 1. Convert Protobuf request to agnos::primitives::Tensor
            std::vector<size_t> input_shape(request->input_shape().begin(), request->input_shape().end());
            std::vector<float> input_data(request->tensor_data().begin(), request->tensor_data().end());
            agnos::primitives::Tensor input_tensor(input_shape, input_data);

            // 2. Run Forward Pass & Measure Latency
            auto start = std::chrono::high_resolution_clock::now();
            agnos::primitives::Tensor output_tensor = network_->forward(input_tensor);
            auto end = std::chrono::high_resolution_clock::now();

            // 3. Populate Protobuf Response
            reply->set_correlation_id(request->correlation_id());
            for (size_t dim : output_tensor.shape()) {
                reply->add_output_shape(dim);
            }
            for (size_t i = 0; i < output_tensor.size(); ++i) {
                reply->add_prediction(output_tensor[i]);
            }
            
            // Record computation time in microseconds
            reply->set_computation_time_us(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

            return grpc::Status::OK;
        } catch (const std::exception& e) {
            reply->set_error_message(e.what());
            return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
        }
    }
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
    
    // Load the model once at startup and inject it into the service
    std::cout << "\nBooting up compute backend and loading global model..." << std::endl;
    std::string model_path = "../../data/models/v1_base_mlp.agnos";
    auto layers = agnos::core::ModelLoader::load_agnos(model_path);
    auto backend = std::make_shared<agnos::backend::CPUBackend>();
    auto nn = std::make_shared<agnos::core::NeuralNetwork>(std::move(layers), backend);

    InferenceEngineServiceImpl service(nn);

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
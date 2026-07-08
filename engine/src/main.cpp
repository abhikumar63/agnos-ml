#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cstdlib> // Needed for std::getenv
#include <fstream> // Needed for file existence check

#include <grpcpp/grpcpp.h>
#include "inference.grpc.pb.h"

#include "core/model_loader.hpp"
#include "backend/cpu_backend.hpp"
#include "core/neural_network.hpp"
#include "primitives/tensor.hpp"
#include <chrono>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using agnos::inference::InferenceEngine;
using agnos::inference::HealthRequest;
using agnos::inference::HealthResponse;
using agnos::inference::PredictRequest;
using agnos::inference::PredictResponse;

// Strictly read from environment configuration, or use a local dev fallback
std::string get_model_path() {
    if (const char* env_path = std::getenv("MODEL_PATH")) {
        if (std::string(env_path) != "") {
            return std::string(env_path);
        }
    }
    return "../../data/models/v1_base_mlp.agnos"; // Local developer fallback
}

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

void RunServer() {
    std::string server_address("0.0.0.0:50051");
    std::string model_path = get_model_path();
    
    std::cout << "\n[INIT] Booting up compute backend..." << std::endl;
    std::cout << "[INIT] Loading global model from: " << model_path << std::endl;
    
    // Will naturally throw std::runtime_error if file is missing, which is caught in main()
    auto layers = agnos::core::ModelLoader::load_agnos(model_path);
    auto backend = std::make_shared<agnos::backend::CPUBackend>();
    auto nn = std::make_shared<agnos::core::NeuralNetwork>(std::move(layers), backend);

    InferenceEngineServiceImpl service(nn);

    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "🚀 [READY] Agnos-ML C++ Compute Engine listening on " << server_address << std::endl;
    
    server->Wait();
}

int main(int argc, char** argv) {
    try {
        RunServer();
    } catch (const std::exception& e) {
        std::cerr << "\n[FATAL] Engine failed to start: " << e.what() << std::endl;
        return EXIT_FAILURE; // Clean exit code to trigger orchestrator restart policies
    }
    return EXIT_SUCCESS;
}
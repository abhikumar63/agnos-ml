#pragma once

#include <vector>
#include <memory>
#include "core/model_loader.hpp"
#include "backend/compute_backend.hpp"

namespace agnos {
namespace core {

class NeuralNetwork {
private:
    std::vector<Layer> layers_;
    std::shared_ptr<agnos::backend::ComputeBackend> backend_;

public:
    // Inject the layers and the chosen hardware backend (Dependency Injection)
    NeuralNetwork(std::vector<Layer> layers, std::shared_ptr<agnos::backend::ComputeBackend> backend);
    
    // Executes the forward pass for inference
    primitives::Tensor forward(const primitives::Tensor& input);
};

} // namespace core
} // namespace agnos
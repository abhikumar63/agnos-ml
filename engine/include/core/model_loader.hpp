#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "primitives/tensor.hpp"

namespace agnos {
namespace core {

using agnos::primitives::Tensor;

// Represents a single parsed layer from the .agnos file
struct Layer {
    Tensor weights;
    Tensor biases;
    uint32_t activation_type; // 0=None, 1=ReLU, 2=Sigmoid, 3=Tanh
};

class ModelLoader {
public:
    // Parses the .agnos binary file and returns a vector of initialized Layers
    static std::vector<Layer> load_agnos(const std::string& filepath);
};

} // namespace core
} // namespace agnos
#include "core/model_loader.hpp"
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <iostream>

namespace agnos {
namespace core {

std::vector<Layer> ModelLoader::load_agnos(const std::string& filepath) {
    // Open file in binary mode
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open model file: " + filepath);
    }

    // 1. Read and validate Magic Number
    char magic[4];
    file.read(magic, 4);
    if (std::strncmp(magic, "AGNS", 4) != 0) {
        throw std::runtime_error("Invalid file format. Magic number mismatch.");
    }

    // 2. Read Global Metadata
    uint32_t version, arch_type, num_layers;
    file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&arch_type), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(uint32_t));

    std::cout << "Loading Model - Version: " << version 
              << ", ArchType: " << arch_type 
              << ", Layers: " << num_layers << std::endl;

    std::vector<Layer> layers;
    layers.reserve(num_layers);

    // 3. Parse each layer
    for (uint32_t i = 0; i < num_layers; ++i) {
        uint32_t layer_type, act_type, in_dim, out_dim;
        
        // Read Layer Metadata
        file.read(reinterpret_cast<char*>(&layer_type), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&act_type), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&in_dim), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&out_dim), sizeof(uint32_t));

        // PyTorch linear weights are exported as (out_features, in_features)
        Tensor weights({in_dim, out_dim});
        Tensor biases({out_dim});

        // Read PyTorch's raw floats into a temporary buffer first (out_dim x in_dim)
        std::vector<float> temp_weights(out_dim * in_dim);
        file.read(reinterpret_cast<char*>(temp_weights.data()), temp_weights.size() * sizeof(float));
        
        // Transpose the matrix mathematically: temp_weights[o][i] -> weights[i][o]
        for (size_t o = 0; o < out_dim; ++o) {
            for (size_t i = 0; i < in_dim; ++i) {
                weights.at(i, o) = temp_weights[o * in_dim + i];
            }
        }

        // Read Raw Floats directly into the Tensor's contiguous memory
        file.read(reinterpret_cast<char*>(biases.mutable_data().data()), biases.size() * sizeof(float));

        layers.push_back({std::move(weights), std::move(biases), act_type});
        
        std::cout << "  -> Loaded Layer " << i << ": Shape(" << in_dim << " -> " << out_dim << "), Act: " << act_type << std::endl;
    }

    return layers;
}

} // namespace core
} // namespace agnos
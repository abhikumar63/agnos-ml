#include "core/neural_network.hpp"
#include <stdexcept>

namespace agnos {
namespace core {

NeuralNetwork::NeuralNetwork(std::vector<Layer> layers, std::shared_ptr<agnos::backend::ComputeBackend> backend)
    : layers_(std::move(layers)), backend_(std::move(backend)) {
    if (!backend_) {
        throw std::invalid_argument("ComputeBackend cannot be null.");
    }
}

primitives::Tensor NeuralNetwork::forward(const primitives::Tensor& input) {
    if (layers_.empty()) {
        throw std::runtime_error("Neural network has no layers loaded.");
    }

    primitives::Tensor current = input;

    // Sequentially push the data through each layer of the network
    for (size_t i = 0; i < layers_.size(); ++i) {
        const auto& layer = layers_[i];

        // 1. Linear Transformation (Matmul)
        current = backend_->matmul(current, layer.weights);

        // 2. Add Biases
        backend_->add_in_place(current, layer.biases);

        // 3. Apply Non-Linear Activation
        switch (layer.activation_type) {
            case 0: // None
                break; 
            case 1: // ReLU
                backend_->relu_in_place(current);
                break;
            case 2: // Sigmoid
                backend_->sigmoid_in_place(current);
                break;
            case 3: // Tanh
                backend_->tanh_in_place(current);
                break;
            default:
                throw std::runtime_error("Unsupported activation type encountered in layer " + std::to_string(i));
        }
    }

    return current;
}

} // namespace core
} // namespace agnos
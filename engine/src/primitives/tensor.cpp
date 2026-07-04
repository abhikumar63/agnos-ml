#include "primitives/tensor.hpp"

namespace agnos {
namespace primitives {

// Allocate contiguous memory based on the product of the shape dimensions
Tensor::Tensor(const std::vector<size_t> &shape) : shape_(shape) {
  if (shape.empty()) {
    throw std::invalid_argument("Tensor shape cannot be empty.");
  }
  size_t total_size = std::accumulate(shape.begin(), shape.end(), 1ULL,
                                      std::multiplies<size_t>());
  data_.resize(total_size, 0.0f); // Zero-initialize the memory
}

// Initialize with existing data (e.g., loading weights from the .agnos file)
Tensor::Tensor(const std::vector<size_t> &shape,
               const std::vector<float> &initial_data)
    : shape_(shape), data_(initial_data) {
  size_t expected_size = std::accumulate(shape.begin(), shape.end(), 1ULL,
                                         std::multiplies<size_t>());
  if (initial_data.size() != expected_size) {
    throw std::invalid_argument(
        "Data size does not match the provided shape dimensions.");
  }
}

size_t Tensor::size() const { return data_.size(); }

float &Tensor::operator[](size_t index) { return data_[index]; }

const float &Tensor::operator[](size_t index) const { return data_[index]; }

// Row-major 2D access: index = (row * total_columns) + column
float &Tensor::at(size_t row, size_t col) {
  if (shape_.size() != 2) {
    throw std::domain_error("at(row, col) can only be used on 2D Tensors.");
  }
  return data_[row * shape_[1] + col];
}

const float &Tensor::at(size_t row, size_t col) const {
  if (shape_.size() != 2) {
    throw std::domain_error("at(row, col) can only be used on 2D Tensors.");
  }
  return data_[row * shape_[1] + col];
}

} // namespace primitives
} // namespace agnos
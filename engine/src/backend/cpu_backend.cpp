#include "backend/cpu_backend.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace agnos {
namespace backend {

Tensor CPUBackend::matmul(const Tensor &a, const Tensor &b) {
  const auto &shape_a = a.shape();
  const auto &shape_b = b.shape();

  // Matrix multiplication rule: Inner dimensions must match.
  // A: (M x K), B: (K x N) -> Result: (M x N)
  if (shape_a.size() != 2 || shape_b.size() != 2) {
    throw std::invalid_argument("Matmul requires 2D tensors.");
  }

  size_t M = shape_a[0];
  size_t K = shape_a[1];
  if (K != shape_b[0]) {
    throw std::invalid_argument("Inner matrix dimensions must agree.");
  }
  size_t N = shape_b[1];

  // Initialize the result tensor with shape (M x N)
  Tensor result({M, N});

  // Standard cache-friendly matrix multiplication (i-j-k loop order)
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (size_t k = 0; k < K; ++k) {
        sum += a.at(i, k) * b.at(k, j);
      }
      result.at(i, j) = sum;
    }
  }

  return result;
}

void CPUBackend::add_in_place(Tensor &a, const Tensor &b) {
  // For bias addition, 'a' is typically (M x N) and 'b' is a bias vector of
  // size (N) We broadcast the vector 'b' across every row of matrix 'a'.

  const auto &shape_a = a.shape();
  const auto &shape_b = b.shape();

  if (shape_a.size() != 2 || shape_b.size() != 1) {
    throw std::invalid_argument(
        "add_in_place expects a 2D tensor and a 1D bias vector.");
  }

  if (shape_a[1] != shape_b[0]) {
    throw std::invalid_argument(
        "Bias vector size must match the number of columns in the matrix.");
  }

  size_t rows = shape_a[0];
  size_t cols = shape_a[1];

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      a.at(i, j) += b[j];
    }
  }
}

// ReLU: f(x) = max(0, x)
void CPUBackend::relu_in_place(Tensor& t) {
    auto& data = t.mutable_data();
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = std::max(0.0f, data[i]);
    }
}

// Sigmoid: f(x) = 1 / (1 + exp(-x))
void CPUBackend::sigmoid_in_place(Tensor& t) {
    auto& data = t.mutable_data();
    for (size_t i = 0; i < data.size(); ++i) {
        // Fast approximation or standard cmath
        data[i] = 1.0f / (1.0f + std::exp(-data[i]));
    }
}

// Tanh: standard hyperbolic tangent
void CPUBackend::tanh_in_place(Tensor& t) {
    auto& data = t.mutable_data();
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = std::tanh(data[i]);
    }
}

} // namespace backend
} // namespace agnos
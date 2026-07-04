#pragma once

#include "primitives/tensor.hpp"
#include <memory>

namespace agnos {
namespace backend {

using agnos::primitives::Tensor;

class ComputeBackend {
public:
  virtual ~ComputeBackend() = default;

  // Core Linear Algebra
  // Computes: C = A * B
  virtual Tensor matmul(const Tensor &a, const Tensor &b) = 0;

  // Computes: C = A + B (element-wise addition, primarily for biases)
  virtual void add_in_place(Tensor &a, const Tensor &b) = 0;
};

} // namespace backend
} // namespace agnos
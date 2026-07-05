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
  virtual Tensor matmul(const Tensor &a, const Tensor &b) = 0;
  virtual void add_in_place(Tensor &a, const Tensor &b) = 0;

  // Activation Functions (In-Place Modifications)
    virtual void relu_in_place(Tensor& t) = 0;
    virtual void sigmoid_in_place(Tensor& t) = 0;
    virtual void tanh_in_place(Tensor& t) = 0;
};

} // namespace backend
} // namespace agnos
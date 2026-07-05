#pragma once

#include "backend/compute_backend.hpp"

namespace agnos {
namespace backend {

class CPUBackend : public ComputeBackend {
public:
  Tensor matmul(const Tensor &a, const Tensor &b) override;
  void add_in_place(Tensor &a, const Tensor &b) override;

  // Activation Functions
    void relu_in_place(Tensor& t) override;
    void sigmoid_in_place(Tensor& t) override;
    void tanh_in_place(Tensor& t) override;
};

} // namespace backend
} // namespace agnos
#pragma once

#include <cstddef>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace agnos {
namespace primitives {

class Tensor {
private:
  std::vector<size_t> shape_;
  std::vector<float> data_; // RAII handles memory lifecycle automatically

public:
  // Constructors
  Tensor() = default;
  explicit Tensor(const std::vector<size_t> &shape);
  Tensor(const std::vector<size_t> &shape,
         const std::vector<float> &initial_data);

  // Rule of Five (Relying on default compiler implementations since std::vector
  // handles deep copies/moves)
  ~Tensor() = default;
  Tensor(const Tensor &) = default;
  Tensor &operator=(const Tensor &) = default;
  Tensor(Tensor &&) noexcept = default;
  Tensor &operator=(Tensor &&) noexcept = default;

  // Accessors
  const std::vector<size_t> &shape() const { return shape_; }
  const std::vector<float> &data() const { return data_; }
  std::vector<float> &mutable_data() { return data_; }

  // Core Utilities
  size_t size() const;

  // Flat memory access
  float &operator[](size_t index);
  const float &operator[](size_t index) const;

  // 2D Matrix convenience accessor
  float &at(size_t row, size_t col);
  const float &at(size_t row, size_t col) const;
};

} // namespace primitives
} // namespace agnos
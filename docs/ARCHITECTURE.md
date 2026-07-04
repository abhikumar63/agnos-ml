# Agnos-ML: Distributed Inference Engine (v2.0)

## Executive Summary
Agnos-ML is an end-to-end, hardware-flexible machine learning inference pipeline. It implements core neural network primitives from scratch in standard C++ and serves them through a high-throughput, horizontally scalable microservice architecture. 

This project demonstrates deep mathematical understanding of AI fundamentals alongside production-grade distributed systems engineering.

---

## 1. System Architecture

The project follows an event-driven microservices architecture, strictly isolating heavy mathematical computation from the ingestion API to ensure non-blocking, low-latency performance.

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Ingestion API** | Go, Fiber | Handles incoming HTTP/gRPC requests, authenticates, and validates payload schema. |
| **Message Broker** | Apache Kafka (KRaft) | Decouples the API from the compute engine, buffering high-volume traffic. Includes a Dead Letter Queue (DLQ). |
| **Compute Engine** | C++20, gRPC | A custom-built, hardware-flexible math engine executing the ML primitives. |
| **State & Caching** | Redis | Caches frequent inference results using tensor-hashed keys. |
| **Persistence** | PostgreSQL | Stores execution metadata, model registry, and system logs. |
| **Observability** | Prometheus | Exposes latency histograms, cache hit rates, and consumer lag. |

---

## 2. Core Engineering Principles & Constraints

### 2.1 Hardware-Flexible `ComputeBackend`
The C++ engine strictly avoids locking into proprietary acceleration for its default state. It defines a `ComputeBackend` abstraction:
* **CPUBackend (Default):** Portable, standard C++ implementation optimized for cache locality. Runs anywhere.
* **CUDABackend (Optional):** Hardware-accelerated implementation added via strategy pattern. 

### 2.2 Separation of Training and Inference
**This system does not train models.** Training occurs offline via Python/PyTorch scripts. The C++ engine is strictly an inference server that loads serialized weights into memory on startup.

### 2.3 Strict Memory Management
The C++ core utilizes strict RAII (Resource Acquisition Is Initialization) and smart pointers. No raw `new`/`delete` calls are permitted during tensor manipulation to prevent memory leaks under heavy load.

---

## 3. The Contract Boundary (gRPC / Protobuf)

The system boundary between the Go API and the C++ Engine is strictly defined by `inference.proto`. 

```proto
syntax = "proto3";
package agnos.inference;

service InferenceEngine {
  rpc Predict(PredictRequest) returns (PredictResponse);
  rpc BatchPredict(BatchPredictRequest) returns (BatchPredictResponse);
  rpc LoadModel(LoadModelRequest) returns (LoadModelResponse);
  rpc HealthCheck(HealthRequest) returns (HealthResponse);
}

message PredictRequest {
  string model_id = 1;
  string correlation_id = 2; 
  repeated int64 input_shape = 3; 
  repeated float tensor_data = 4; 
}

message PredictResponse {
  string correlation_id = 1;
  repeated float prediction = 2; 
  repeated int64 output_shape = 3; 
  int64 computation_time_us = 4; 
  string error_message = 5; 
}
```

---

## 4. The `.agnos` Binary Serialization Format

To bypass heavy dependencies like ONNX or LibTorch, Agnos-ML uses a highly optimized, custom binary format to load neural network weights into C++ memory. 

All data is stored in **Little-Endian** byte order.

### 4.1 File Header (16 Bytes)
| Offset | Size | Type    | Description |
| :---   | :--- | :---    | :--- |
| 0      | 4    | char[4] | Magic Number: "AGNS" |
| 4      | 4    | uint32  | Version (e.g., 1) |
| 8      | 4    | uint32  | Architecture Type (0 = MLP) |
| 12     | 4    | uint32  | Number of Layers (N) |

### 4.2 Layer Metadata (Repeated N times)
| Size | Type   | Description |
| :--- | :---   | :--- |
| 4    | uint32 | Layer Type (0 = Dense/Linear) |
| 4    | uint32 | Activation Type (0=None, 1=ReLU, 2=Sigmoid, 3=Tanh) |
| 4    | uint32 | Input Dimension (In_Features) |
| 4    | uint32 | Output Dimension (Out_Features) |

### 4.3 Raw Weights & Biases (Repeated N times)
| Size | Type    | Description |
| :--- | :---    | :--- |
| (In * Out) * 4 | float32 | Weight Matrix (Flattened, row-major) |
| Out * 4        | float32 | Bias Vector |

---

## 5. Caching & Observability Strategy

### Redis Tensor Caching
Naively caching floating-point tensors results in near-zero hit rates. Agnos-ML implements a hashed caching strategy in the Go API:
`Cache Key = Hash(model_id + MD5(input_tensor_rounded_to_4_decimal_places))`

### Prometheus Metrics
The Go API exposes a `/metrics` endpoint tracking:
1.  **Inference Latency:** Histograms for p50, p95, and p99.
2.  **Queue Health:** Kafka consumer lag.
3.  **Efficiency:** Redis cache hit/miss ratio.
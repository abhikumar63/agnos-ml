# Agnos-ML

Agnos-ML is a high-performance, containerized Machine Learning inference platform. It is designed with a strict separation of concerns, utilizing a lightning-fast C++ compute engine for heavy tensor operations and a highly concurrent Go API Gateway to handle client traffic.

## 🏗️ Architecture

The system is composed of two primary microservices communicating over a local Docker network:

1. **C++ Compute Engine (`/engine`)**
   * **Role:** Core inference execution.
   * **Stack:** C++17, gRPC, Protobuf, Alpine Linux.
   * **Features:** Loads custom binary `.agnos` neural network models. Implements a custom CPU backend for matrix multiplications (MatMul) and non-linear activations (ReLU, Sigmoid, Tanh). Exposed strictly via a high-performance gRPC server on port `50051`.

2. **Go API Gateway (`/backend`)**
   * **Role:** Client-facing REST API and orchestrator.
   * **Stack:** Go 1.22+, Fiber (HTTP framework), gRPC Client.
   * **Features:** Ingests HTTP JSON payloads, validates shapes, maps them to Protobuf messages, and routes them to the C++ Engine. Handles request timeouts and connection pooling. Exposed on port `8080`.

## 🚀 Getting Started

### Prerequisites
* Docker
* Docker Compose

### Building and Running
The entire stack is containerized and orchestrated via `docker-compose`.

```bash
# Clone the repository
git clone [https://github.com/abhikumar63/agnos-ml.git](https://github.com/abhikumar63/agnos-ml.git)
cd agnos-ml

# Build the containers and launch the network
docker-compose up --build
```

### Volume Mounts & Models
The C++ Engine requires a compiled `.agnos` model file to boot. 
Place your model in the `data/models/` directory at the root of the project. Docker Compose automatically mounts this directory to `/app/data/models/` inside the C++ container to bypass macOS file-sharing restrictions.

## 📡 API Endpoints

### 1. Health Check
Checks the status of both the Go REST API and the internal C++ gRPC Engine.

**Request:** `GET /health`

**Response:**
```json
{
  "api_status": "UP",
  "engine_status": "SERVING"
}
```

### 2. Predict
Pushes a tensor through the neural network.

**Request:** `POST /predict`
```json
{
  "model_id": "v1_base_mlp",
  "correlation_id": "req-12345",
  "input_shape": [1, 20],
  "tensor_data": [1.0, 0.5, -0.2, 0.1, 0.0]
}
```

**Response:**
```json
{
  "correlation_id": "req-12345",
  "output_shape": [1, 2],
  "prediction": [0.89, 0.11],
  "compute_time_us": 26
}
```

## 🛠️ Roadmap / Next Steps
* [ ] **Caching Layer:** Implement Redis in the Go API to hash and cache frequent tensor requests.
* [ ] **Telemetry:** Add Prometheus `/metrics` scraping to the Go API.
* [ ] **GPU Acceleration:** Extend the C++ backend to support CUDA.
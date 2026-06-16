#!/usr/bin/env bash

# Fail immediately if any command exits with a non-zero status
set -e

# Forcefully add the default Go binary paths to this script's execution context
export PATH="$PATH:$HOME/go/bin:$(go env GOPATH)/bin"

echo "🚀 Starting Protobuf Compilation for Agnos-ML..."

# 1. Define absolute paths to ensure the script can be run from anywhere
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

GO_OUT_DIR="$ROOT_DIR/backend/internal/proto_gen"
CPP_OUT_DIR="$ROOT_DIR/engine/src/proto_gen"

# 2. Verify Prerequisites
if ! command -v protoc &> /dev/null; then
    echo "❌ Error: 'protoc' is not installed."
    echo "Install via: brew install protobuf (Mac) or apt-get install protobuf-compiler (Linux)"
    exit 1
fi

if ! command -v protoc-gen-go &> /dev/null || ! command -v protoc-gen-go-grpc &> /dev/null; then
    echo "❌ Error: Go protobuf plugins are missing."
    echo "Run: go install google.golang.org/protobuf/cmd/protoc-gen-go@latest"
    echo "Run: go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest"
    exit 1
fi

if ! command -v grpc_cpp_plugin &> /dev/null; then
    echo "❌ Error: 'grpc_cpp_plugin' is not installed."
    echo "This is required for C++ gRPC generation. Install gRPC for your system."
    exit 1
fi

# 3. Clean and recreate output directories to prevent zombie files
echo "🧹 Cleaning old generated files..."
rm -rf "$GO_OUT_DIR" "$CPP_OUT_DIR"
mkdir -p "$GO_OUT_DIR" "$CPP_OUT_DIR"

# 4. Compile Go Stubs
echo "🐹 Generating Go stubs..."
protoc -I="$SCRIPT_DIR" \
    --go_out="$GO_OUT_DIR" --go_opt=paths=source_relative \
    --go-grpc_out="$GO_OUT_DIR" --go-grpc_opt=paths=source_relative \
    "$SCRIPT_DIR/inference.proto"

# 5. Compile C++ Stubs
echo "⚙️  Generating C++ stubs..."
protoc -I="$SCRIPT_DIR" \
    --cpp_out="$CPP_OUT_DIR" \
    --grpc_out="$CPP_OUT_DIR" \
    --plugin=protoc-gen-grpc="$(which grpc_cpp_plugin)" \
    "$SCRIPT_DIR/inference.proto"

echo "✅ Protobuf compilation complete!"
echo "   Go files at:  $GO_OUT_DIR"
echo "   C++ files at: $CPP_OUT_DIR"
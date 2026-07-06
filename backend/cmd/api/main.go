package main

import (
	"context"
	"log"
	"time"

	pb "agnos-backend/proto_gen" // Imports your generated Protobuf stubs

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/recover"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// HTTP JSON payload structure
type PredictPayload struct {
	ModelID       string    `json:"model_id"`
	CorrelationID string    `json:"correlation_id"`
	InputShape    []int64   `json:"input_shape"`
	TensorData    []float32 `json:"tensor_data"`
}

func main() {
	// 1. Establish gRPC connection to the C++ Compute Engine
	engineAddress := "localhost:50051"
	conn, err := grpc.Dial(engineAddress, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("❌ Failed to connect to C++ Engine: %v", err)
	}
	defer conn.Close()

	grpcClient := pb.NewInferenceEngineClient(conn)
	log.Println("✅ Successfully connected to C++ Engine at", engineAddress)

	// 2. Initialize Fiber app
	app := fiber.New(fiber.Config{
		AppName: "Agnos-ML API Gateway v2.0",
	})
	app.Use(recover.New())
	app.Use(logger.New())

	// 3. Health Route
	app.Get("/health", func(c *fiber.Ctx) error {
		// Ping the C++ Engine as part of the health check
		ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
		defer cancel()

		engineStatus := "DOWN"
		res, err := grpcClient.HealthCheck(ctx, &pb.HealthRequest{})
		if err == nil && res.Status == pb.HealthResponse_SERVING {
			engineStatus = "SERVING"
		}

		return c.JSON(fiber.Map{
			"api_status":    "UP",
			"engine_status": engineStatus,
		})
	})

	// 4. Inference Route
	app.Post("/predict", func(c *fiber.Ctx) error {
		payload := new(PredictPayload)
		if err := c.BodyParser(payload); err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "Invalid JSON payload format"})
		}

		// Map HTTP JSON -> Protobuf Request
		req := &pb.PredictRequest{
			ModelId:       payload.ModelID,
			CorrelationId: payload.CorrelationID,
			InputShape:    payload.InputShape,
			TensorData:    payload.TensorData,
		}

		// Enforce a strict timeout so a stalled C++ engine doesn't hang the HTTP client
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()

		// Call the C++ Engine over gRPC
		res, err := grpcClient.Predict(ctx, req)
		if err != nil {
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": "Compute Engine Error: " + err.Error()})
		}

		// Map Protobuf Response -> HTTP JSON
		return c.JSON(fiber.Map{
			"correlation_id":  res.CorrelationId,
			"prediction":      res.Prediction,
			"output_shape":    res.OutputShape,
			"compute_time_us": res.ComputationTimeUs,
		})
	})

	log.Println("🚀 Starting Agnos-ML Go API Gateway on port 8080...")
	log.Fatal(app.Listen(":8080"))
}

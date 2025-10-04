package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/qdrant-recommended/goapp/go/pkg/qdrant"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	client, err := qdrant.NewClientFromEnv()
	if err != nil {
		log.Fatalf("create client: %v", err)
	}

	if err := client.EnsureCollection(ctx, 768); err != nil {
		log.Fatalf("ensure collection: %v", err)
	}

	fmt.Println("Collection ready. Example query payload:")

	samplePoints := []qdrant.Point{
		{
			ID:     1,
			Vector: []float32{0.1, 0.2, 0.3},
			Payload: map[string]any{
				"category": "education",
			},
		},
	}

	data, _ := json.MarshalIndent(samplePoints, "", "  ")
	fmt.Println(string(data))
}

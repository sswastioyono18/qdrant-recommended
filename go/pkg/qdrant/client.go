package qdrant

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"
)

// Client wraps access to the Qdrant HTTP API. It mirrors the small helper
// utilities that exist in the Python codebase so that the Go port can provide
// equivalent behaviour.
type Client struct {
	httpClient *http.Client
	baseURL    string
	collection string
}

// NewClientFromEnv constructs a client using the same environment variables the
// Python helper relies on. Defaults are aligned to `qdrant_store.get_client`.
func NewClientFromEnv() (*Client, error) {
	host := getenvDefault("QDRANT_HOST", "localhost")
	port := getenvDefault("QDRANT_PORT", "6333")
	collection := getenvDefault("QDRANT_COLLECTION", "campaigns_demo")

	if _, err := strconv.Atoi(port); err != nil {
		return nil, fmt.Errorf("invalid QDRANT_PORT value %q: %w", port, err)
	}

	baseURL := fmt.Sprintf("http://%s:%s", host, port)

	return &Client{
		httpClient: &http.Client{Timeout: 10 * time.Second},
		baseURL:    strings.TrimRight(baseURL, "/"),
		collection: collection,
	}, nil
}

func getenvDefault(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// EnsureCollection checks whether the configured collection exists and creates
// it if necessary. The behaviour mirrors `ensure_collection` from the Python
// code and defaults to the cosine distance metric.
func (c *Client) EnsureCollection(ctx context.Context, vectorSize int) error {
	if vectorSize <= 0 {
		return errors.New("vector size must be positive")
	}

	exists, err := c.collectionExists(ctx)
	if err != nil {
		return err
	}
	if exists {
		return nil
	}

	payload := map[string]any{
		"vectors": map[string]any{
			"size":     vectorSize,
			"distance": "Cosine",
		},
	}

	req, err := c.newRequest(ctx, http.MethodPut, c.collectionPath(), payload)
	if err != nil {
		return err
	}

	return c.doAndCheck(req, http.StatusAccepted)
}

// UpsertCampaigns mirrors `upsert_campaigns` from the Python implementation by
// pushing the provided points into Qdrant using the `/points` endpoint.
func (c *Client) UpsertCampaigns(ctx context.Context, points []Point) error {
	if len(points) == 0 {
		return errors.New("no points supplied")
	}

	payload := map[string]any{"points": points}
	req, err := c.newRequest(ctx, http.MethodPut, c.collectionPath("points"), payload)
	if err != nil {
		return err
	}

	return c.doAndCheck(req, http.StatusAccepted)
}

// QuerySimilar wraps the search endpoint to find points similar to the provided
// vector, applying any optional filters. The logic mirrors the helper in
// `qdrant_store.query_similar` by translating simple equality filters to a
// `should` clause so that any of the supplied conditions may match.
func (c *Client) QuerySimilar(ctx context.Context, vector []float32, topK int, filters map[string]any) ([]SearchResult, error) {
	if len(vector) == 0 {
		return nil, errors.New("query vector cannot be empty")
	}
	if topK <= 0 {
		topK = 50
	}

	body := searchRequest{
		Vector:      vector,
		Limit:       topK,
		WithPayload: true,
	}

	if len(filters) > 0 {
		filter := Filter{}
		for k, v := range filters {
			filter.Should = append(filter.Should, FieldCondition{
				Key: k,
				Match: MatchValue{
					Value: v,
				},
			})
		}
		body.Filter = &filter
	}

	req, err := c.newRequest(ctx, http.MethodPost, c.collectionPath("points", "search"), body)
	if err != nil {
		return nil, err
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, decodeAPIError(resp)
	}

	var payload searchResponse
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, fmt.Errorf("decode search response: %w", err)
	}

	return payload.Result, nil
}

func (c *Client) collectionExists(ctx context.Context) (bool, error) {
	req, err := c.newRequest(ctx, http.MethodGet, c.collectionPath(), nil)
	if err != nil {
		return false, err
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusOK:
		return true, nil
	case http.StatusNotFound:
		return false, nil
	default:
		return false, decodeAPIError(resp)
	}
}

func (c *Client) collectionPath(parts ...string) string {
	escaped := url.PathEscape(c.collection)
	if len(parts) == 0 {
		return fmt.Sprintf("/collections/%s", escaped)
	}
	return fmt.Sprintf("/collections/%s/%s", escaped, strings.Join(parts, "/"))
}

func (c *Client) newRequest(ctx context.Context, method, path string, body any) (*http.Request, error) {
	var reader io.Reader
	if body != nil {
		buf := &bytes.Buffer{}
		if err := json.NewEncoder(buf).Encode(body); err != nil {
			return nil, fmt.Errorf("encode request body: %w", err)
		}
		reader = buf
	}

	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, reader)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	return req, nil
}

func (c *Client) doAndCheck(req *http.Request, expected int) error {
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != expected {
		return decodeAPIError(resp)
	}
	return nil
}

// decodeAPIError extracts the error returned by Qdrant and wraps it in a Go
// error so that callers can react appropriately.
func decodeAPIError(resp *http.Response) error {
	data, _ := io.ReadAll(resp.Body)
	var payload struct {
		Status  string `json:"status"`
		Result  any    `json:"result"`
		Error   string `json:"error"`
		Message string `json:"message"`
	}
	if len(data) > 0 {
		_ = json.Unmarshal(data, &payload)
	}

	msg := payload.Error
	if msg == "" {
		msg = payload.Message
	}
	if msg == "" {
		msg = strings.TrimSpace(string(data))
	}
	if msg == "" {
		msg = resp.Status
	}

	return fmt.Errorf("qdrant API error (%d): %s", resp.StatusCode, msg)
}

// Point models the payload required by the `/points` endpoint. It supports both
// integer and string identifiers by using an empty interface for the `ID`.
type Point struct {
	ID      any            `json:"id"`
	Vector  []float32      `json:"vector"`
	Payload map[string]any `json:"payload,omitempty"`
}

// searchRequest mirrors the JSON payload expected by the search endpoint.
type searchRequest struct {
	Vector      []float32 `json:"vector"`
	Limit       int       `json:"limit,omitempty"`
	Filter      *Filter   `json:"filter,omitempty"`
	WithPayload bool      `json:"with_payload"`
}

// Filter and related structs provide a minimal representation of the Qdrant
// filtering DSL. Only the fields required by the migration are implemented.
type Filter struct {
	Should []FieldCondition `json:"should,omitempty"`
}

type FieldCondition struct {
	Key   string     `json:"key"`
	Match MatchValue `json:"match"`
}

type MatchValue struct {
	Value any `json:"value"`
}

// SearchResult represents a single point returned by Qdrant.
type SearchResult struct {
	ID      json.RawMessage `json:"id"`
	Score   float64         `json:"score"`
	Payload map[string]any  `json:"payload"`
	Vector  []float32       `json:"vector,omitempty"`
}

type searchResponse struct {
	Result []SearchResult `json:"result"`
}

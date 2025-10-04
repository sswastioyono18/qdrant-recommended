# Go Migration Plan

This document captures the initial plan for migrating the Python-based Qdrant recommendation helpers to Go.

## Goals

1. Provide a Go module that mirrors the high-level responsibilities of `src/app/qdrant_store.py` and related helpers.
2. Allow the existing Python code and the new Go implementation to coexist while the migration is in progress.
3. Document the step-by-step strategy so that additional modules (e.g. recommendation pipelines, embedding utilities) can be ported incrementally.

## Current Status

- `go.mod` initialised with module `github.com/qdrant-recommended/goapp`.
- `go/pkg/qdrant/client.go` implements equivalents of `get_client`, `ensure_collection`, `upsert_campaigns`, and `query_similar` using the Qdrant HTTP API.
- `go/cmd/demo/main.go` demonstrates client construction and shows how to marshal sample points ready for upsert calls.

## Next Steps

1. **Repository structure**:
   - Keep the Go code under `go/` during migration to avoid disrupting the Python tooling.
   - Once parity is achieved, promote the Go module to the repository root and retire the Python entrypoints.

2. **Feature parity**:
   - Port data models from `models.py` and `user_prefs_store.py` to typed Go structs.
   - Recreate the recommendation workflows (`recommend.py`, `comprehensive_recommender.py`) in Go, calling into the new `qdrant` package.
   - Implement embedding generation either by calling an external service or binding to an existing Go embedding library.

3. **Testing**:
   - Introduce unit tests for the Go client using httptest to validate request/response behaviour.
   - Mirror the scenarios covered by the Python test harness (where available) to ensure behaviour matches.

4. **Operational readiness**:
   - Provide CLI entrypoints (e.g. `cmd/recommender`) that mirror the existing scripts.
   - Document environment variables, configuration, and deployment steps for the Go binaries.

5. **Decommission Python code**:
   - Once each component is ported and validated, remove the Python counterpart and update documentation accordingly.

This staged approach keeps the migration incremental and testable while giving immediate value through the reusable Go client.

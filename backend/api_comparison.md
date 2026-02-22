# Backend API — Framework Comparison

## Flask vs FastAPI for HNDSR

| Dimension | Flask | FastAPI | Winner |
|-----------|-------|---------|--------|
| **Async support** | Requires extensions (Quart) | Native `async/await` | ✅ FastAPI |
| **Request throughput** | ~500 req/s (sync) | ~3000 req/s (async) | ✅ FastAPI |
| **Auto-documentation** | Manual (Swagger via extension) | Built-in OpenAPI + Swagger | ✅ FastAPI |
| **Type validation** | Manual or Marshmallow | Built-in via Pydantic | ✅ FastAPI |
| **GPU inference** | Blocks event loop | `await loop.run_in_executor()` | ✅ FastAPI |
| **WebSocket support** | Flask-SocketIO (complex) | Native WebSocket | ✅ FastAPI |
| **Community maturity** | 15+ years, huge ecosystem | 5+ years, growing fast | ⚖️ Tie |
| **Learning curve** | Simple, minimal boilerplate | Slightly steeper (Pydantic) | ⚖️ Tie |
| **Production battle-tested** | Very mature | Proven at scale (Netflix, Uber) | ⚖️ Tie |

## Decision: **FastAPI**

### Why FastAPI for GPU ML serving:

1. **Async is mandatory**: GPU inference takes 500ms–3s per request. A synchronous framework would block the entire server during each inference, limiting throughput to 1 concurrent request per worker.

2. **Pydantic validation**: Input images need validation (size, format, dimensions) before touching the GPU. Flask requires manual validation code; FastAPI validates automatically via request schemas.

3. **Background tasks**: Long-running inferences can be offloaded to background tasks or Redis queues. FastAPI's `BackgroundTasks` makes this seamless.

4. **Health checks matter**: Kubernetes liveness/readiness probes need fast responses even while the GPU is busy. FastAPI's async architecture lets `/health` respond instantly without waiting for `/infer` to complete.

### What Flask would be better for:
- Simple REST APIs without GPU (Flask is simpler for CRUD apps)
- Projects requiring Flask-specific extensions (Flask-Admin, Flask-Login)
- Teams with deep Flask expertise who value simplicity over performance

# Memory Intelligence Service

Zero-LLM memory optimization using local ML models for the Cogniz Memory Platform.

## Overview

This Python FastAPI service provides semantic analysis of memories without using LLMs:
- **Duplicate Detection** - Identifies similar memories using sentence-transformers
- **Clustering** - Groups related memories using DBSCAN
- **Priority Scoring** - Ranks memories by importance
- **Consolidation** - Optimizes memory storage

## Features

- 🚀 **Zero-LLM** - Uses local embedding models (no API costs)
- ⚡ **Fast** - All-MiniLM-L6-v2 model is only 22MB
- 🎯 **Accurate** - Cosine similarity for duplicate detection
- 📊 **Scalable** - Batch processing support
- 🔒 **Private** - All processing happens locally

## Architecture

```
WordPress Plugin → REST API → FastAPI Service → ML Models
                                    ↓
                            sentence-transformers
                                    ↓
                              DBSCAN clustering
```

## Quick Start

### Local Development

1. **Clone repository**
```bash
git clone https://github.com/cognizonline/cogniz-memory-intelligence.git
cd cogniz-memory-intelligence
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run service**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

5. **Test health endpoint**
```bash
curl http://localhost:8000/health
```

### Production Deployment

See [RENDER_DEPLOYMENT_GUIDE.md](RENDER_DEPLOYMENT_GUIDE.md) for complete deployment instructions to Render.com.

## API Endpoints

### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "database": "connected"
}
```

### Duplicate Detection (WordPress Integration)
```http
POST /api/v1/analyze/duplicates
Content-Type: application/json

{
  "memories": [
    {"id": "mem_123", "content": "Sample memory content"},
    {"id": "mem_124", "content": "Very similar memory content"}
  ],
  "threshold": 0.90
}
```

**Response:**
```json
{
  "duplicates": [
    {
      "memory_id": "mem_124",
      "duplicate_of": "mem_123",
      "similarity": 0.95
    }
  ]
}
```

### Clustering (WordPress Integration)
```http
POST /api/v1/analyze/clusters
Content-Type: application/json

{
  "memories": [
    {"id": "mem_123", "content": "Machine learning topic"},
    {"id": "mem_124", "content": "AI and ML discussion"},
    {"id": "mem_125", "content": "Python programming"}
  ],
  "sensitivity": 0.70
}
```

**Response:**
```json
{
  "clusters": [
    {
      "cluster_id": "cluster_0",
      "memory_ids": ["mem_123", "mem_124"],
      "representative_id": "mem_123",
      "size": 2
    }
  ]
}
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | `8000` |
| `DB_HOST` | Database host (optional) | `localhost` |
| `DB_USER` | Database user (optional) | `root` |
| `DB_PASSWORD` | Database password (optional) | - |
| `DB_NAME` | Database name (optional) | `wordpress` |

### WordPress Integration

1. Install WordPress plugin: `memory-platform-direct`
2. Navigate to **Memory Platform > Dashboard > Intelligence**
3. Configure Service URL: `https://your-service.onrender.com`
4. Enable duplicate detection and clustering
5. Click "Analyze Now" to test

## Technology Stack

- **FastAPI** - Modern Python web framework
- **sentence-transformers** - Semantic embeddings (all-MiniLM-L6-v2)
- **scikit-learn** - DBSCAN clustering, TF-IDF
- **NumPy** - Numerical operations
- **Uvicorn** - ASGI server

## Performance

- **Model Size:** 22MB (all-MiniLM-L6-v2)
- **Inference Speed:** ~50 memories/second
- **Memory Usage:** ~200MB RAM
- **Cold Start:** 2-3 seconds (model loading)

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black app/
```

### Type Checking

```bash
mypy app/
```

## Deployment

### Render.com (Recommended)

See [RENDER_DEPLOYMENT_GUIDE.md](RENDER_DEPLOYMENT_GUIDE.md)

### Docker

```bash
docker build -t memory-intelligence .
docker run -p 8000:8000 memory-intelligence
```

### Heroku

```bash
heroku create cogniz-intelligence
git push heroku main
```

## Roadmap

- [x] Duplicate detection
- [x] Clustering
- [x] WordPress integration
- [ ] Batch optimization
- [ ] Memory consolidation
- [ ] Priority scoring
- [ ] Analytics dashboard
- [ ] Multi-language support

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

## License

MIT License - see LICENSE file

## Support

- **Documentation:** [RENDER_DEPLOYMENT_GUIDE.md](RENDER_DEPLOYMENT_GUIDE.md)
- **Issues:** https://github.com/cognizonline/cogniz-memory-intelligence/issues
- **Email:** support@cogniz.online

## Acknowledgments

- [sentence-transformers](https://www.sbert.net/) - Semantic text embeddings
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [scikit-learn](https://scikit-learn.org/) - Machine learning library

---

**Version:** 1.0.0
**Last Updated:** November 14, 2025

# ML Serving Platform

A production-grade Machine Learning model serving platform built with FastAPI, AWS S3, Docker, Prometheus, Grafana, and Terraform.

## Tech Stack

- **API**: FastAPI + Uvicorn
- **ML**: PyTorch + EfficientNet-B0
- **Storage**: AWS S3 (model storage)
- **Infrastructure**: Terraform (IaC)
- **Monitoring**: Prometheus + Grafana
- **Containerization**: Docker + Docker Compose
- **CI/CD**: GitHub Actions

## Architecture

```
Client → FastAPI → EfficientNet-B0 → Predictions
              ↓
         AWS S3 (model storage)
              ↓
        Prometheus (metrics)
              ↓
         Grafana (dashboard)
              ↓
        Terraform (AWS infra)
```

## Features

- Image classification via EfficientNet-B0
- Model loading from AWS S3
- Real-time Prometheus metrics
- Grafana monitoring dashboard
- Dockerized for easy deployment
- Terraform IaC for AWS resources
- GitHub Actions CI/CD pipeline

## Getting Started

### With Docker Compose
```bash
docker-compose up --build
```

### Local Development
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Run Tests
```bash
pytest tests/ -v
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root endpoint |
| GET | `/health` | Health check |
| POST | `/predict` | Image classification |
| POST | `/load-model` | Load model from S3 |
| GET | `/metrics` | Prometheus metrics |
| GET | `/docs` | Swagger UI |

## Example Usage

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

## Monitoring

- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (admin/admin)
- Metrics: `http://localhost:8000/metrics`

## Infrastructure (Terraform)

```bash
cd terraform
terraform init
terraform plan
terraform apply
```

## Environment Variables

```env
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1
S3_BUCKET=ml-models-bucket-vishal
MODEL_KEY=models/classifier.pt
```
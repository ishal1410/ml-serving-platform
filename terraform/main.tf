terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# S3 bucket for ML models
resource "aws_s3_bucket" "ml_models" {
  bucket = var.s3_bucket_name
  tags = {
    Name        = "ML Models Bucket"
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "ml_models" {
  bucket = aws_s3_bucket.ml_models.id
  versioning_configuration {
    status = "Enabled"
  }
}

# ECR repository for Docker images
resource "aws_ecr_repository" "ml_serving" {
  name                 = "ml-serving-platform"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Environment = var.environment
  }
}
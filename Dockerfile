variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "s3_bucket_name" {
  description = "S3 bucket for ML models"
  type        = string
  default     = "ml-models-bucket-vishal"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "development"
}
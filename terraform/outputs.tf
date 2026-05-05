output "s3_bucket_name" {
  description = "S3 bucket name"
  value       = aws_s3_bucket.ml_models.id
}

output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.ml_serving.repository_url
}
variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type (g4dn.xlarge=T4 16GB, g5.xlarge=A10G 24GB)"
  type        = string
  default     = "g4dn.xlarge"
  validation {
    condition     = contains(["g4dn.xlarge", "g4dn.2xlarge", "g5.xlarge", "g5.2xlarge", "p3.2xlarge"], var.instance_type)
    error_message = "Must be a GPU instance type."
  }
}

variable "use_spot" {
  description = "Use spot instance (50-70% cheaper, can be interrupted)"
  type        = bool
  default     = false
}

variable "spot_max_price" {
  description = "Maximum spot price per hour"
  type        = string
  default     = "0.30"
}

variable "root_volume_size_gb" {
  description = "Root EBS volume size in GB"
  type        = number
  default     = 120
}

variable "key_name" {
  description = "Name for EC2 SSH key pair"
  type        = string
  default     = "phi4-training-key"
}

variable "ssh_public_key" {
  description = "SSH public key content for instance access"
  type        = string
  sensitive   = true
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed for SSH and API access"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Restrict to your IP in production
}

variable "create_s3_bucket" {
  description = "Whether to create an S3 bucket for model artifacts"
  type        = bool
  default     = true
}

variable "s3_bucket_name" {
  description = "S3 bucket name prefix (account ID will be appended)"
  type        = string
  default     = "phi4-training-artifacts"
}

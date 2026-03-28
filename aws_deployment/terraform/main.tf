# =============================================================================
# main.tf — Phi-4 EC2 Training Infrastructure
# Alternative to launch_ec2.sh using Terraform
#
# Usage:
#   terraform init
#   terraform plan
#   terraform apply
#   terraform destroy  # After training is complete
# =============================================================================

terraform {
  required_version = ">= 1.6"
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

# ─────────────────────────────────────────────────────────────────────────────
# Data: Latest Deep Learning AMI
# ─────────────────────────────────────────────────────────────────────────────
data "aws_ami" "dlami" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu*22.04*"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

# ─────────────────────────────────────────────────────────────────────────────
# Networking
# ─────────────────────────────────────────────────────────────────────────────
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

resource "aws_security_group" "phi4" {
  name        = "phi4-training-sg"
  description = "Phi-4 training security group"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "SSH from allowed IPs"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  ingress {
    description = "API port"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.common_tags
}

# ─────────────────────────────────────────────────────────────────────────────
# IAM Role
# ─────────────────────────────────────────────────────────────────────────────
resource "aws_iam_role" "phi4" {
  name = "phi4-training-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "cloudwatch" {
  role       = aws_iam_role.phi4.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
}

resource "aws_iam_role_policy_attachment" "s3" {
  role       = aws_iam_role.phi4.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_iam_role_policy_attachment" "secrets" {
  role       = aws_iam_role.phi4.name
  policy_arn = "arn:aws:iam::aws:policy/SecretsManagerReadWrite"
}

resource "aws_iam_instance_profile" "phi4" {
  name = "phi4-training-profile"
  role = aws_iam_role.phi4.name
}

# ─────────────────────────────────────────────────────────────────────────────
# SSH Key
# ─────────────────────────────────────────────────────────────────────────────
resource "aws_key_pair" "phi4" {
  key_name   = var.key_name
  public_key = var.ssh_public_key
  tags       = local.common_tags
}

# ─────────────────────────────────────────────────────────────────────────────
# EC2 Instance
# ─────────────────────────────────────────────────────────────────────────────
resource "aws_instance" "phi4" {
  ami                    = data.aws_ami.dlami.id
  instance_type          = var.instance_type
  key_name               = aws_key_pair.phi4.key_name
  vpc_security_group_ids = [aws_security_group.phi4.id]
  iam_instance_profile   = aws_iam_instance_profile.phi4.name
  subnet_id              = data.aws_subnets.default.ids[0]

  # Use spot if enabled
  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        max_price                      = var.spot_max_price
        instance_interruption_behavior = "terminate"
        spot_instance_type             = "one-time"
      }
    }
  }

  root_block_device {
    volume_size           = var.root_volume_size_gb
    volume_type           = "gp3"
    iops                  = 3000
    throughput            = 125
    delete_on_termination = false
    encrypted             = true
  }

  user_data = base64encode(templatefile("${path.module}/user_data.sh.tpl", {
    github_repo = "https://github.com/eduardd76/fine-tuned-Phi-4-14B.git"
    setup_script = "aws_deployment/setup_instance.sh"
  }))

  tags = merge(local.common_tags, {
    Name         = "phi4-training-${formatdate("YYYYMMDD-HHmm", timestamp())}"
    AutoShutdown = "true"
  })

  lifecycle {
    ignore_changes = [tags["Name"]]
  }
}

# CloudWatch auto-shutdown alarm (20h idle)
resource "aws_cloudwatch_metric_alarm" "auto_shutdown" {
  alarm_name          = "phi4-autoshutdown-${aws_instance.phi4.id}"
  alarm_description   = "Auto-shutdown after 20h low CPU (training complete)"
  namespace           = "AWS/EC2"
  metric_name         = "CPUUtilization"
  statistic           = "Average"
  period              = 3600
  evaluation_periods  = 20
  threshold           = 5
  comparison_operator = "LessThanThreshold"
  treat_missing_data  = "notBreaching"

  dimensions = {
    InstanceId = aws_instance.phi4.id
  }

  alarm_actions = [
    "arn:aws:swf:${var.aws_region}:${data.aws_caller_identity.current.account_id}:action/actions/AWS_EC2.InstanceId.Stop/1.0"
  ]
}

data "aws_caller_identity" "current" {}

# ─────────────────────────────────────────────────────────────────────────────
# S3 Bucket for artifacts
# ─────────────────────────────────────────────────────────────────────────────
resource "aws_s3_bucket" "phi4_artifacts" {
  count  = var.create_s3_bucket ? 1 : 0
  bucket = "${var.s3_bucket_name}-${data.aws_caller_identity.current.account_id}"
  tags   = local.common_tags
}

resource "aws_s3_bucket_lifecycle_configuration" "phi4_artifacts" {
  count  = var.create_s3_bucket ? 1 : 0
  bucket = aws_s3_bucket.phi4_artifacts[0].id

  rule {
    id     = "archive-old-checkpoints"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "GLACIER"
    }

    filter {
      prefix = "checkpoints/"
    }
  }
}

# ─────────────────────────────────────────────────────────────────────────────
# Locals
# ─────────────────────────────────────────────────────────────────────────────
locals {
  common_tags = {
    Project     = "phi4-network-architect"
    Environment = "training"
    ManagedBy   = "terraform"
  }
}

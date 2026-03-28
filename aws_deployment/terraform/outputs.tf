output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.phi4.id
}

output "public_ip" {
  description = "Public IP address"
  value       = aws_instance.phi4.public_ip
}

output "public_dns" {
  description = "Public DNS name"
  value       = aws_instance.phi4.public_dns
}

output "ssh_command" {
  description = "SSH command to connect"
  value       = "ssh -i ~/.ssh/${var.key_name}.pem ubuntu@${aws_instance.phi4.public_ip}"
}

output "ami_used" {
  description = "AMI used for the instance"
  value       = data.aws_ami.dlami.id
}

output "s3_bucket" {
  description = "S3 bucket for artifacts"
  value       = var.create_s3_bucket ? aws_s3_bucket.phi4_artifacts[0].id : "not created"
}

output "cost_estimate" {
  description = "Estimated training cost"
  value = {
    hourly_rate    = var.instance_type == "g4dn.xlarge" ? "$0.526" : "$1.006"
    training_hours = "15"
    total_estimate = var.instance_type == "g4dn.xlarge" ? "~$7.89" : "~$15.09"
    with_spot      = var.use_spot ? "70% cheaper with spot" : "on-demand pricing"
  }
}

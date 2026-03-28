#!/usr/bin/env bash
# =============================================================================
# launch_ec2.sh — Launch EC2 instance for Phi-4 fine-tuning
#
# Usage:
#   ./launch_ec2.sh                       # g4dn.xlarge (T4, 16GB) — default
#   INSTANCE_TYPE=g5.xlarge ./launch_ec2.sh  # A10G 24GB
#   USE_SPOT=true ./launch_ec2.sh         # Spot instance (~70% cheaper)
#
# Requirements: AWS CLI v2, jq
# Cost estimates (on-demand):
#   g4dn.xlarge: $0.526/hr  → 15h training ≈ $7.89
#   g5.xlarge:   $1.006/hr  → 15h training ≈ $15.09
#   Spot saves 50-70% of on-demand cost
# =============================================================================
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Configuration (override with env vars)
# ─────────────────────────────────────────────────────────────────────────────
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g4dn.xlarge}"
USE_SPOT="${USE_SPOT:-false}"
SPOT_MAX_PRICE="${SPOT_MAX_PRICE:-0.30}"          # Max hourly spot price
VOLUME_SIZE="${VOLUME_SIZE:-120}"                  # GB
KEY_NAME="${KEY_NAME:-phi4-training-key}"
SG_NAME="${SG_NAME:-phi4-training-sg}"
INSTANCE_NAME="phi4-training-$(date +%Y%m%d-%H%M)"
GITHUB_REPO="https://github.com/eduardd76/fine-tuned-Phi-4-14B.git"

# Deep Learning AMI (Ubuntu 22.04) — updated regularly, find latest:
# aws ec2 describe-images --owners amazon --filters "Name=name,Values=Deep Learning AMI GPU PyTorch*Ubuntu 22.04*" --query 'Images[0].ImageId'
AMI_ID="${AMI_ID:-}"  # Will auto-detect if empty

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

command -v aws &>/dev/null || die "AWS CLI not installed. Install: https://aws.amazon.com/cli/"
command -v jq  &>/dev/null || die "jq not installed. Install: sudo apt-get install jq"

aws sts get-caller-identity &>/dev/null || die "AWS credentials not configured. Run: aws configure"

log "=== Phi-4 EC2 Launch ==="
log "  Region:        $REGION"
log "  Instance type: $INSTANCE_TYPE"
log "  Spot mode:     $USE_SPOT"
log "  Volume:        ${VOLUME_SIZE}GB"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Auto-detect latest Deep Learning AMI
# ─────────────────────────────────────────────────────────────────────────────
if [[ -z "$AMI_ID" ]]; then
    log "Finding latest Deep Learning AMI (Ubuntu 22.04)..."
    AMI_ID=$(aws ec2 describe-images \
        --region "$REGION" \
        --owners amazon \
        --filters \
            "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu*22.04*" \
            "Name=state,Values=available" \
        --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
        --output text)

    # Fallback to standard DLAMI if OSS not found
    if [[ -z "$AMI_ID" || "$AMI_ID" == "None" ]]; then
        AMI_ID=$(aws ec2 describe-images \
            --region "$REGION" \
            --owners amazon \
            --filters \
                "Name=name,Values=Deep Learning AMI GPU PyTorch*Ubuntu 22.04*" \
                "Name=state,Values=available" \
            --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
            --output text)
    fi
    [[ -n "$AMI_ID" && "$AMI_ID" != "None" ]] || die "Could not find Deep Learning AMI. Set AMI_ID manually."
    log "  Using AMI: $AMI_ID"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 2. Create SSH key pair (if not exists)
# ─────────────────────────────────────────────────────────────────────────────
KEY_FILE="${HOME}/.ssh/${KEY_NAME}.pem"
if ! aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" &>/dev/null; then
    log "Creating SSH key pair: $KEY_NAME"
    mkdir -p ~/.ssh
    aws ec2 create-key-pair \
        --key-name "$KEY_NAME" \
        --region "$REGION" \
        --query 'KeyMaterial' \
        --output text > "$KEY_FILE"
    chmod 400 "$KEY_FILE"
    log "  Key saved: $KEY_FILE"
else
    log "  Using existing key: $KEY_NAME"
    [[ -f "$KEY_FILE" ]] || log "  WARNING: Key file not found at $KEY_FILE — you may need it for SSH"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3. Create Security Group (if not exists)
# ─────────────────────────────────────────────────────────────────────────────
MY_IP=$(curl -sf https://checkip.amazonaws.com || echo "0.0.0.0/0")
MY_IP="${MY_IP}/32"

SG_ID=$(aws ec2 describe-security-groups \
    --region "$REGION" \
    --filters "Name=group-name,Values=$SG_NAME" \
    --query 'SecurityGroups[0].GroupId' \
    --output text 2>/dev/null || echo "None")

if [[ "$SG_ID" == "None" || -z "$SG_ID" ]]; then
    log "Creating security group: $SG_NAME"
    VPC_ID=$(aws ec2 describe-vpcs \
        --region "$REGION" \
        --filters "Name=isDefault,Values=true" \
        --query 'Vpcs[0].VpcId' \
        --output text)

    SG_ID=$(aws ec2 create-security-group \
        --region "$REGION" \
        --group-name "$SG_NAME" \
        --description "Phi-4 training security group" \
        --vpc-id "$VPC_ID" \
        --query 'GroupId' \
        --output text)

    # SSH from your IP only
    aws ec2 authorize-security-group-ingress \
        --region "$REGION" \
        --group-id "$SG_ID" \
        --protocol tcp --port 22 --cidr "$MY_IP"

    # API port (optional — from your IP)
    aws ec2 authorize-security-group-ingress \
        --region "$REGION" \
        --group-id "$SG_ID" \
        --protocol tcp --port 8000 --cidr "$MY_IP"

    # TensorBoard (SSH tunnel only, but port open for convenience)
    aws ec2 authorize-security-group-ingress \
        --region "$REGION" \
        --group-id "$SG_ID" \
        --protocol tcp --port 6006 --cidr "$MY_IP"

    log "  Security group created: $SG_ID (SSH from $MY_IP)"
else
    log "  Using existing security group: $SG_ID"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. Create IAM role for CloudWatch + S3 (idempotent)
# ─────────────────────────────────────────────────────────────────────────────
IAM_ROLE_NAME="phi4-training-role"
IAM_PROFILE_NAME="phi4-training-profile"

if ! aws iam get-role --role-name "$IAM_ROLE_NAME" &>/dev/null; then
    log "Creating IAM role: $IAM_ROLE_NAME"
    aws iam create-role \
        --role-name "$IAM_ROLE_NAME" \
        --assume-role-policy-document '{
            "Version":"2012-10-17",
            "Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]
        }' > /dev/null

    aws iam attach-role-policy --role-name "$IAM_ROLE_NAME" \
        --policy-arn "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
    aws iam attach-role-policy --role-name "$IAM_ROLE_NAME" \
        --policy-arn "arn:aws:iam::aws:policy/AmazonS3FullAccess"
    aws iam attach-role-policy --role-name "$IAM_ROLE_NAME" \
        --policy-arn "arn:aws:iam::aws:policy/SecretsManagerReadWrite"

    aws iam create-instance-profile \
        --instance-profile-name "$IAM_PROFILE_NAME" > /dev/null
    aws iam add-role-to-instance-profile \
        --instance-profile-name "$IAM_PROFILE_NAME" \
        --role-name "$IAM_ROLE_NAME"

    sleep 10  # IAM propagation delay
    log "  IAM role created"
else
    log "  Using existing IAM role: $IAM_ROLE_NAME"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 5. User data script (runs on first boot)
# ─────────────────────────────────────────────────────────────────────────────
USER_DATA=$(base64 -w0 << 'USERDATA'
#!/bin/bash
set -e
LOG=/var/log/phi4-setup.log
echo "=== Phi-4 Boot Setup $(date) ===" >> $LOG

# Install CloudWatch agent
apt-get install -y amazon-cloudwatch-agent >> $LOG 2>&1

# Signal CloudWatch that instance is running
/opt/aws/bin/cfn-signal --success true --region us-east-1 || true

# Clone repo
cd /home/ubuntu
sudo -u ubuntu git clone https://github.com/eduardd76/fine-tuned-Phi-4-14B.git >> $LOG 2>&1

# Run setup
sudo -u ubuntu bash /home/ubuntu/fine-tuned-Phi-4-14B/aws_deployment/setup_instance.sh >> $LOG 2>&1

echo "=== Boot setup complete $(date) ===" >> $LOG
USERDATA
)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Launch instance
# ─────────────────────────────────────────────────────────────────────────────
EBS_CONFIG=$(cat <<EOF
[{
    "DeviceName": "/dev/sda1",
    "Ebs": {
        "VolumeSize": $VOLUME_SIZE,
        "VolumeType": "gp3",
        "Iops": 3000,
        "Throughput": 125,
        "DeleteOnTermination": false,
        "Encrypted": true
    }
}]
EOF
)

if [[ "$USE_SPOT" == "true" ]]; then
    log "Launching SPOT instance: $INSTANCE_TYPE"
    LAUNCH_RESULT=$(aws ec2 run-instances \
        --region "$REGION" \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SG_ID" \
        --iam-instance-profile "Name=$IAM_PROFILE_NAME" \
        --block-device-mappings "$EBS_CONFIG" \
        --user-data "$USER_DATA" \
        --instance-market-options "{\"MarketType\":\"spot\",\"SpotOptions\":{\"MaxPrice\":\"$SPOT_MAX_PRICE\",\"SpotInstanceType\":\"one-time\",\"InstanceInterruptionBehavior\":\"terminate\"}}" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME},{Key=Project,Value=phi4-training},{Key=AutoShutdown,Value=true}]" \
        --output json)
else
    log "Launching ON-DEMAND instance: $INSTANCE_TYPE"
    LAUNCH_RESULT=$(aws ec2 run-instances \
        --region "$REGION" \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SG_ID" \
        --iam-instance-profile "Name=$IAM_PROFILE_NAME" \
        --block-device-mappings "$EBS_CONFIG" \
        --user-data "$USER_DATA" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME},{Key=Project,Value=phi4-training},{Key=AutoShutdown,Value=true}]" \
        --output json)
fi

INSTANCE_ID=$(echo "$LAUNCH_RESULT" | jq -r '.Instances[0].InstanceId')
log "  Instance ID: $INSTANCE_ID"

# ─────────────────────────────────────────────────────────────────────────────
# 7. Wait for instance to be running
# ─────────────────────────────────────────────────────────────────────────────
log "Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

PUBLIC_DNS=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicDnsName' \
    --output text)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Set up auto-shutdown CloudWatch alarm
# ─────────────────────────────────────────────────────────────────────────────
log "Setting up auto-shutdown alarm (20h timeout)..."
aws cloudwatch put-metric-alarm \
    --region "$REGION" \
    --alarm-name "phi4-autoshutdown-${INSTANCE_ID}" \
    --alarm-description "Auto-shutdown Phi-4 training instance after 20 hours" \
    --namespace "AWS/EC2" \
    --metric-name "CPUUtilization" \
    --dimensions "Name=InstanceId,Value=$INSTANCE_ID" \
    --statistic Average \
    --period 3600 \
    --evaluation-periods 20 \
    --threshold 5 \
    --comparison-operator LessThanThreshold \
    --alarm-actions "arn:aws:swf:${REGION}:$(aws sts get-caller-identity --query Account --output text):action/actions/AWS_EC2.InstanceId.Stop/1.0" \
    --treat-missing-data notBreaching || log "  WARNING: Could not create CloudWatch alarm (need CloudWatch Alarm role)"

# ─────────────────────────────────────────────────────────────────────────────
# 9. Save connection details
# ─────────────────────────────────────────────────────────────────────────────
CONN_FILE="./connection_details.txt"
cat > "$CONN_FILE" << EOF
=== Phi-4 EC2 Instance Details ===
Instance ID:    $INSTANCE_ID
Instance Type:  $INSTANCE_TYPE
Public IP:      $PUBLIC_IP
Public DNS:     $PUBLIC_DNS
Region:         $REGION
Key File:       $KEY_FILE
Launch Time:    $(date)

=== SSH Commands ===
# Connect:
ssh -i $KEY_FILE ubuntu@$PUBLIC_IP

# TensorBoard tunnel:
ssh -i $KEY_FILE -L 6006:localhost:6006 ubuntu@$PUBLIC_IP

# API tunnel:
ssh -i $KEY_FILE -L 8000:localhost:8000 ubuntu@$PUBLIC_IP

=== After connecting ===
# Start training:
cd ~/fine-tuned-Phi-4-14B
screen -S training
./run_full_pipeline.sh

# Monitor:
screen -r training
# Detach: Ctrl+A D

=== Termination ===
# Terminate instance (after training):
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION

# Create EBS snapshot first:
EBS_ID=\$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --region $REGION --query 'Reservations[0].Instances[0].BlockDeviceMappings[0].Ebs.VolumeId' --output text)
aws ec2 create-snapshot --volume-id \$EBS_ID --description "phi4-training-final" --region $REGION
EOF

log ""
log "=== Launch Complete ==="
log "  Instance: $INSTANCE_ID"
log "  IP:       $PUBLIC_IP"
log "  DNS:      $PUBLIC_DNS"
log ""
log "Connection details saved to: $CONN_FILE"
log ""
log "Connect with:"
log "  ssh -i $KEY_FILE ubuntu@$PUBLIC_IP"
log ""
log "Note: Wait 3-5 minutes for setup_instance.sh to complete before running pipeline."
log "      Check setup progress: ssh ubuntu@$PUBLIC_IP 'tail -f /var/log/phi4-setup.log'"

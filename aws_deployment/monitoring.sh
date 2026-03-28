#!/usr/bin/env bash
# =============================================================================
# monitoring.sh — CloudWatch + TensorBoard monitoring for Phi-4 training
#
# Usage:
#   ./aws_deployment/monitoring.sh <EC2_IP>
#   ./aws_deployment/monitoring.sh ec2-xx-xx-xx.compute.amazonaws.com
# =============================================================================
set -euo pipefail

EC2_IP="${1:-localhost}"
KEY_FILE="${KEY_FILE:-~/.ssh/phi4-training-key.pem}"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

echo "=== Phi-4 Training Monitor ==="
echo "  EC2: $EC2_IP"
echo "  Region: $REGION"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Option 1: TensorBoard via SSH tunnel
# ─────────────────────────────────────────────────────────────────────────────
start_tensorboard_tunnel() {
    log "Starting TensorBoard SSH tunnel..."
    log "  Open: http://localhost:6006"

    # Kill existing tunnel
    pkill -f "ssh.*6006:localhost:6006" 2>/dev/null || true

    ssh -i "$KEY_FILE" -N -L 6006:localhost:6006 "ubuntu@$EC2_IP" &
    TB_PID=$!
    log "  SSH tunnel PID: $TB_PID (kill to stop)"

    # Wait and open browser
    sleep 2
    if command -v xdg-open &>/dev/null; then
        xdg-open http://localhost:6006 2>/dev/null || true
    elif command -v open &>/dev/null; then
        open http://localhost:6006 2>/dev/null || true
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Option 2: Watch training log in real-time
# ─────────────────────────────────────────────────────────────────────────────
watch_training_log() {
    log "Streaming training log from $EC2_IP..."
    ssh -i "$KEY_FILE" "ubuntu@$EC2_IP" \
        "tail -f /data/logs/pipeline.log 2>/dev/null || tail -f /var/log/phi4-setup.log"
}

# ─────────────────────────────────────────────────────────────────────────────
# Option 3: CloudWatch dashboard query
# ─────────────────────────────────────────────────────────────────────────────
query_cloudwatch() {
    log "CloudWatch training metrics (last 1 hour)..."

    # Get instance ID from IP
    INSTANCE_ID=$(aws ec2 describe-instances \
        --region "$REGION" \
        --filters "Name=ip-address,Values=$EC2_IP" \
        --query 'Reservations[0].Instances[0].InstanceId' \
        --output text 2>/dev/null || echo "")

    if [[ -z "$INSTANCE_ID" || "$INSTANCE_ID" == "None" ]]; then
        log "  Could not find instance ID for $EC2_IP"
        return
    fi

    log "  Instance: $INSTANCE_ID"

    # CPU utilization
    aws cloudwatch get-metric-statistics \
        --region "$REGION" \
        --namespace AWS/EC2 \
        --metric-name CPUUtilization \
        --dimensions "Name=InstanceId,Value=$INSTANCE_ID" \
        --start-time "$(date -u -d '1 hour ago' '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || date -u -v-1H '+%Y-%m-%dT%H:%M:%SZ')" \
        --end-time "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
        --period 300 \
        --statistics Average \
        --query 'sort_by(Datapoints, &Timestamp)[-5:].Average' \
        --output table 2>/dev/null || log "  CloudWatch metrics not available yet"

    # Recent pipeline log
    log "  Recent pipeline log:"
    aws logs get-log-events \
        --region "$REGION" \
        --log-group-name "/phi4/training" \
        --log-stream-name "pipeline" \
        --limit 20 \
        --query 'events[].message' \
        --output text 2>/dev/null | tail -10 || log "  Log group not yet created"
}

# ─────────────────────────────────────────────────────────────────────────────
# Option 4: Quick GPU status via SSH
# ─────────────────────────────────────────────────────────────────────────────
quick_gpu_status() {
    log "GPU status on $EC2_IP:"
    ssh -i "$KEY_FILE" "ubuntu@$EC2_IP" \
        "nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | awk -F, '{printf \"  GPU: %s | Util: %s%% | Mem: %s/%sMB | Temp: %s°C\n\", \$1, \$2, \$4, \$5, \$6}'" \
        2>/dev/null || log "  Could not connect to $EC2_IP"
}

# ─────────────────────────────────────────────────────────────────────────────
# Menu
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$EC2_IP" != "localhost" ]]; then
    echo "Options:"
    echo "  1) Stream training log"
    echo "  2) GPU status"
    echo "  3) CloudWatch metrics"
    echo "  4) TensorBoard tunnel"
    echo "  5) All (background)"
    echo ""
    read -rp "Choose [1-5]: " CHOICE

    case "$CHOICE" in
        1) watch_training_log ;;
        2) quick_gpu_status ;;
        3) query_cloudwatch ;;
        4) start_tensorboard_tunnel; wait ;;
        5)
            quick_gpu_status
            query_cloudwatch
            start_tensorboard_tunnel
            watch_training_log
            ;;
        *) log "Invalid choice" ;;
    esac
else
    # Local monitoring
    query_cloudwatch
fi

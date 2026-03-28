"""
CloudWatch metrics, dashboard and alarm setup for Phi-4 Network Architect.

Usage:
    python monitoring/cloudwatch_setup.py --setup-all
    python monitoring/cloudwatch_setup.py --metrics-only
    python monitoring/cloudwatch_setup.py --dashboard-only
    python monitoring/cloudwatch_setup.py --alarms-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import boto3

NAMESPACE = "Phi4NetworkArchitect"
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
INSTANCE_ID = os.environ.get("EC2_INSTANCE_ID", "")
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")
DASHBOARD_NAME = "Phi4-Network-Architect"

cw = boto3.client("cloudwatch", region_name=REGION)


# ---------------------------------------------------------------------------
# Custom metric helpers
# ---------------------------------------------------------------------------

def put_metric(metric_name: str, value: float, unit: str = "None", dimensions: list | None = None) -> None:
    """Publish a single custom metric to CloudWatch."""
    cw.put_metric_data(
        Namespace=NAMESPACE,
        MetricData=[
            {
                "MetricName": metric_name,
                "Dimensions": dimensions or [],
                "Value": value,
                "Unit": unit,
            }
        ],
    )


def publish_inference_metrics(latency_ms: float, confidence: float, has_think: bool) -> None:
    """Called from the API after each inference — publishes latency/confidence."""
    put_metric("InferenceLatency", latency_ms, "Milliseconds")
    put_metric("ConfidenceScore", confidence, "None")
    put_metric("ThinkBlockPresent", 1.0 if has_think else 0.0, "None")


def publish_request_metrics(endpoint: str, status_code: int) -> None:
    """Called from API middleware — publishes per-endpoint counters."""
    dims = [{"Name": "Endpoint", "Value": endpoint}]
    put_metric("RequestCount", 1.0, "Count", dims)
    if status_code >= 400:
        put_metric("ErrorCount", 1.0, "Count", dims)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

DASHBOARD_BODY = {
    "widgets": [
        {
            "type": "metric",
            "x": 0, "y": 0, "width": 12, "height": 6,
            "properties": {
                "title": "Inference Latency (ms)",
                "metrics": [
                    [NAMESPACE, "InferenceLatency", {"stat": "p50", "label": "p50"}],
                    [NAMESPACE, "InferenceLatency", {"stat": "p95", "label": "p95"}],
                    [NAMESPACE, "InferenceLatency", {"stat": "p99", "label": "p99"}],
                ],
                "period": 300,
                "view": "timeSeries",
            },
        },
        {
            "type": "metric",
            "x": 12, "y": 0, "width": 12, "height": 6,
            "properties": {
                "title": "Confidence Score (avg)",
                "metrics": [
                    [NAMESPACE, "ConfidenceScore", {"stat": "Average", "label": "avg confidence"}],
                ],
                "period": 300,
                "yAxis": {"left": {"min": 0, "max": 1}},
                "view": "timeSeries",
            },
        },
        {
            "type": "metric",
            "x": 0, "y": 6, "width": 12, "height": 6,
            "properties": {
                "title": "Request Count & Errors",
                "metrics": [
                    [NAMESPACE, "RequestCount", {"stat": "Sum", "label": "total requests", "color": "#2ca02c"}],
                    [NAMESPACE, "ErrorCount", {"stat": "Sum", "label": "errors", "color": "#d62728"}],
                ],
                "period": 300,
                "view": "timeSeries",
            },
        },
        {
            "type": "metric",
            "x": 12, "y": 6, "width": 12, "height": 6,
            "properties": {
                "title": "<think> Block Rate",
                "metrics": [
                    [NAMESPACE, "ThinkBlockPresent", {"stat": "Average", "label": "% with think block"}],
                ],
                "period": 300,
                "yAxis": {"left": {"min": 0, "max": 1}},
                "view": "timeSeries",
            },
        },
        # EC2 system metrics
        {
            "type": "metric",
            "x": 0, "y": 12, "width": 12, "height": 6,
            "properties": {
                "title": "EC2 CPU Utilization",
                "metrics": [
                    ["AWS/EC2", "CPUUtilization", "InstanceId", INSTANCE_ID or "REPLACE_ME",
                     {"stat": "Average", "label": "CPU %"}],
                ],
                "period": 60,
                "view": "timeSeries",
            },
        },
        {
            "type": "metric",
            "x": 12, "y": 12, "width": 12, "height": 6,
            "properties": {
                "title": "EC2 Network I/O",
                "metrics": [
                    ["AWS/EC2", "NetworkIn", "InstanceId", INSTANCE_ID or "REPLACE_ME",
                     {"stat": "Sum", "label": "bytes in"}],
                    ["AWS/EC2", "NetworkOut", "InstanceId", INSTANCE_ID or "REPLACE_ME",
                     {"stat": "Sum", "label": "bytes out"}],
                ],
                "period": 60,
                "view": "timeSeries",
            },
        },
    ]
}


def create_dashboard() -> None:
    print(f"Creating CloudWatch dashboard: {DASHBOARD_NAME}")
    cw.put_dashboard(
        DashboardName=DASHBOARD_NAME,
        DashboardBody=json.dumps(DASHBOARD_BODY),
    )
    print(f"  Dashboard URL: https://{REGION}.console.aws.amazon.com/cloudwatch/home"
          f"?region={REGION}#dashboards:name={DASHBOARD_NAME}")


# ---------------------------------------------------------------------------
# Alarms
# ---------------------------------------------------------------------------

def _alarm_actions() -> list[str]:
    return [SNS_TOPIC_ARN] if SNS_TOPIC_ARN else []


def create_alarms() -> None:
    alarms = [
        {
            "AlarmName": "Phi4-HighLatency",
            "AlarmDescription": "p95 inference latency > 15 seconds for 5 minutes",
            "MetricName": "InferenceLatency",
            "Namespace": NAMESPACE,
            "Statistic": "p95",
            "Period": 300,
            "EvaluationPeriods": 1,
            "Threshold": 15_000,
            "ComparisonOperator": "GreaterThanThreshold",
        },
        {
            "AlarmName": "Phi4-LowConfidence",
            "AlarmDescription": "Average confidence score < 0.70 for 10 minutes",
            "MetricName": "ConfidenceScore",
            "Namespace": NAMESPACE,
            "Statistic": "Average",
            "Period": 600,
            "EvaluationPeriods": 1,
            "Threshold": 0.70,
            "ComparisonOperator": "LessThanThreshold",
        },
        {
            "AlarmName": "Phi4-HighErrorRate",
            "AlarmDescription": "More than 10 errors in 5 minutes",
            "MetricName": "ErrorCount",
            "Namespace": NAMESPACE,
            "Statistic": "Sum",
            "Period": 300,
            "EvaluationPeriods": 1,
            "Threshold": 10,
            "ComparisonOperator": "GreaterThanThreshold",
        },
        {
            "AlarmName": "Phi4-LowThinkBlockRate",
            "AlarmDescription": "Think block rate < 80% — model may be degraded",
            "MetricName": "ThinkBlockPresent",
            "Namespace": NAMESPACE,
            "Statistic": "Average",
            "Period": 900,
            "EvaluationPeriods": 2,
            "Threshold": 0.80,
            "ComparisonOperator": "LessThanThreshold",
        },
    ]

    for alarm in alarms:
        alarm["AlarmActions"] = _alarm_actions()
        print(f"  Creating alarm: {alarm['AlarmName']}")
        cw.put_metric_alarm(**alarm)


# ---------------------------------------------------------------------------
# CloudWatch agent config (GPU metrics via collectd)
# ---------------------------------------------------------------------------

AGENT_CONFIG = {
    "metrics": {
        "namespace": NAMESPACE,
        "metrics_collected": {
            "cpu": {"measurement": ["cpu_usage_idle", "cpu_usage_user", "cpu_usage_system"],
                    "metrics_collection_interval": 60},
            "disk": {"measurement": ["used_percent"], "resources": ["/", "/data"],
                     "metrics_collection_interval": 60},
            "mem": {"measurement": ["mem_used_percent"], "metrics_collection_interval": 60},
        },
    },
    "logs": {
        "logs_collected": {
            "files": {
                "collect_list": [
                    {"file_path": "/data/logs/phi4-api.log",
                     "log_group_name": "/phi4/api",
                     "log_stream_name": "{instance_id}/api"},
                    {"file_path": "/data/logs/training.log",
                     "log_group_name": "/phi4/training",
                     "log_stream_name": "{instance_id}/training"},
                ]
            }
        }
    },
}


def write_agent_config(path: str = "/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json") -> None:
    print(f"Writing CloudWatch agent config to {path}")
    with open(path, "w") as f:
        json.dump(AGENT_CONFIG, f, indent=2)
    print("  Restart agent: sudo systemctl restart amazon-cloudwatch-agent")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Set up CloudWatch for Phi-4 Network Architect")
    parser.add_argument("--setup-all", action="store_true")
    parser.add_argument("--dashboard-only", action="store_true")
    parser.add_argument("--alarms-only", action="store_true")
    parser.add_argument("--write-agent-config", action="store_true")
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(0)

    if args.setup_all or args.dashboard_only:
        create_dashboard()

    if args.setup_all or args.alarms_only:
        print("Creating CloudWatch alarms...")
        create_alarms()
        print("Done.")

    if args.write_agent_config:
        write_agent_config()


if __name__ == "__main__":
    main()

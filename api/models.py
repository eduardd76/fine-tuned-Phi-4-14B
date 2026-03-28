"""Pydantic models for Phi-4 API requests and responses."""

from __future__ import annotations

import uuid
from typing import Any
from pydantic import BaseModel, Field


class NetworkRequirements(BaseModel):
    users: int | None = Field(None, description="Number of users/endpoints")
    sites: int = Field(1, description="Number of geographic sites")
    compliance: list[str] = Field(
        default_factory=list,
        description="Compliance standards: PCI-DSS, HIPAA, NIST CSF, ISO 27001",
        examples=[["PCI-DSS", "HIPAA"]],
    )
    uptime: float = Field(99.9, description="Required uptime percentage", ge=90.0, le=100.0)
    budget: str = Field("flexible", description="Budget constraint: tight/moderate/flexible/unlimited")
    data_center: bool = Field(False, description="Includes data center design")
    wireless: bool = Field(True, description="Requires wireless coverage")
    wan_type: str | None = Field(None, description="WAN preference: SD-WAN/MPLS/internet/hybrid")
    cloud_connectivity: bool = Field(False, description="Requires cloud (AWS/Azure/GCP) connectivity")
    redundancy_level: str = Field("N+1", description="Redundancy: N+1/N+2/active-active")

    class Config:
        extra = "allow"  # Allow additional requirement fields
        json_schema_extra = {
            "example": {
                "users": 5000,
                "sites": 20,
                "compliance": ["PCI-DSS", "HIPAA"],
                "uptime": 99.99,
                "budget": "flexible",
                "data_center": True,
                "wan_type": "SD-WAN",
            }
        }


class DesignRequest(BaseModel):
    requirements: NetworkRequirements
    max_tokens: int = Field(2048, ge=256, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=1.0)
    use_cache: bool = True

    class Config:
        json_schema_extra = {
            "example": {
                "requirements": {
                    "users": 5000,
                    "sites": 20,
                    "compliance": ["PCI-DSS"],
                    "uptime": 99.99,
                }
            }
        }


class TroubleshootRequest(BaseModel):
    symptom: str = Field(..., min_length=10, description="Description of the network issue")
    device: str = Field("unknown", description="Affected device hostname or IP")
    protocol: str | None = Field(None, description="Affected protocol: BGP/OSPF/VLAN/etc")
    topology_context: str | None = Field(None, description="Relevant topology information")
    logs: str | None = Field(None, description="Relevant syslog or debug output (max 2000 chars)")
    max_tokens: int = Field(2048, ge=256, le=4096)

    class Config:
        json_schema_extra = {
            "example": {
                "symptom": "BGP session flapping between PE and CE, notification sent every 10 minutes",
                "device": "pe-router-01",
                "protocol": "BGP",
            }
        }


class EstimateRequest(BaseModel):
    requirements: NetworkRequirements
    include_opex: bool = True
    include_implementation: bool = True
    currency: str = Field("USD", pattern="^[A-Z]{3}$")
    max_tokens: int = Field(2048, ge=256, le=4096)


class CostBreakdown(BaseModel):
    capex: float | None = None
    annual_opex: float | None = None
    implementation: float | None = None
    currency: str = "USD"
    notes: str = ""


class DesignResponse(BaseModel):
    reasoning: str = Field(description="Chain-of-thought reasoning in <think> tags")
    design: str = Field(description="Final network architecture recommendation")
    cost_estimate: CostBreakdown
    sources: list[str] = Field(description="Referenced standards and design guides")
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float
    requires_human_review: bool
    model: str = "phi4-network-architect"

    class Config:
        json_schema_extra = {
            "example": {
                "reasoning": "<think>Step 1: Analyze scale...</think>",
                "design": "Three-tier architecture recommended...",
                "cost_estimate": {"capex": 9300000, "annual_opex": 1860000, "currency": "USD"},
                "sources": ["CCNP Enterprise Design ENSLD", "PCI-DSS 4.0.1"],
                "confidence": 0.95,
                "latency_ms": 2340.5,
                "requires_human_review": False,
            }
        }


class TroubleshootResponse(BaseModel):
    reasoning: str
    diagnosis: str
    remediation_steps: list[str]
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float
    severity: str = "unknown"


class EstimateResponse(BaseModel):
    reasoning: str
    breakdown: str
    cost_estimate: CostBreakdown
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float

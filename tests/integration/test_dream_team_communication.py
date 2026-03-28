"""Integration tests for Dream Team Virtual Architect communication."""
from __future__ import annotations
import asyncio, pytest
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.fixture
def mock_engine():
    engine = MagicMock()
    from dataclasses import dataclass
    @dataclass
    class FakeResult:
        reasoning = "Step 1: Analyze scale..."
        answer = "Three-tier architecture recommended."
        sources = ["CCNP Enterprise Design ENSLD"]
        confidence = 0.92
        latency_ms = 1234.0
        has_think_block = True
        reasoning_steps = 3
        raw = "<think>Step 1: Analyze scale...</think>Three-tier architecture recommended."
    engine.infer.return_value = FakeResult()
    return engine

@pytest.mark.asyncio
async def test_receive_design_task(mock_engine):
    with patch("dream_team_integration.virtual_architect_agent.Phi4InferenceEngine", return_value=mock_engine):
        from dream_team_integration.virtual_architect_agent import VirtualArchitectAgent
        agent = VirtualArchitectAgent(model_path="/tmp/fake-model")
        agent._engine = mock_engine
        task = {"type": "design", "context": {"requirements": {"users": 500, "sites": 3}}, "priority": "medium"}
        result = await agent.receive_task(task)
        assert result["status"] == "success"
        assert result["confidence"] > 0.0
        assert "<think>" in result["reasoning"]

@pytest.mark.asyncio
async def test_receive_troubleshoot_task(mock_engine):
    from dream_team_integration.virtual_architect_agent import VirtualArchitectAgent
    agent = VirtualArchitectAgent(model_path="/tmp/fake-model")
    agent._engine = mock_engine
    task = {"type": "troubleshoot", "context": {"symptom": "BGP neighbor down", "device": "router-1"}}
    result = await agent.receive_task(task)
    assert result["status"] == "success"

@pytest.mark.asyncio
async def test_receive_estimate_task(mock_engine):
    from dream_team_integration.virtual_architect_agent import VirtualArchitectAgent
    agent = VirtualArchitectAgent(model_path="/tmp/fake-model")
    agent._engine = mock_engine
    task = {"type": "estimate", "context": {"users": 1000, "sites": 5}}
    result = await agent.receive_task(task)
    assert result["status"] == "success"

def test_get_capabilities():
    from dream_team_integration.virtual_architect_agent import VirtualArchitectAgent, CAPABILITIES
    agent = VirtualArchitectAgent(model_path="/tmp/fake")
    caps = agent.get_capabilities()
    assert "network_topology_design" in caps
    assert len(caps) >= 6

@pytest.mark.asyncio
async def test_error_handling(mock_engine):
    from dream_team_integration.virtual_architect_agent import VirtualArchitectAgent
    agent = VirtualArchitectAgent(model_path="/tmp/fake-model")
    mock_engine.infer.side_effect = RuntimeError("GPU OOM")
    agent._engine = mock_engine
    result = await agent.receive_task({"type": "design", "context": {}})
    assert result["status"] == "error"
    assert result["confidence"] == 0.0
    assert result["requires_human_approval"] is True

def test_parse_phi4_output():
    from dream_team_integration.phi4_inference import parse_phi4_output
    raw = "<think>Step 1: Analyze scale.\nStep 2: Select topology.</think>\nThree-tier is best. See [CCNP Enterprise Design ENSLD] and RFC 4271."
    result = parse_phi4_output(raw, model_name="phi4", latency_ms=500.0)
    assert result.has_think_block
    assert result.reasoning_steps == 2
    assert "CCNP Enterprise Design ENSLD" in result.sources
    assert "RFC 4271" in result.sources
    assert result.confidence > 0.5
    assert "Three-tier" in result.answer

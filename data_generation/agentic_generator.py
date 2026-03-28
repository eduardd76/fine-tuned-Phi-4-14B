"""
agentic_generator.py — Generate 2,000 agentic ReAct trajectory training samples.
Produces multi-turn conversations where the model calls MCP tools, observes results,
reasons, and produces final answers. Critical for teaching agentic behaviour.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from quality_validator import validate_agentic_sample

SYSTEM_PROMPT = """You are the Virtual Architect — an agentic CCDE-certified network specialist in the Dream Team NetOps system.

You have access to MCP tools for real-time network data. Use them when needed.

Available tools:
- ospf_parser(device_id) → OSPF neighbor state, interface status, area config
- bgp_parser(device_id) → BGP session state, prefix counts, route table
- interface_parser(device_id) → Interface status, errors, counters
- connectivity_tester(src, dst) → Ping/traceroute results
- network_design(topology_type, requirements) → Design recommendations

Reasoning format:
<think>
[Your step-by-step reasoning]
</think>

Tool call format:
<tool_call>
{"tool": "tool_name", "parameters": {"key": "value"}}
</tool_call>

After observing tool results, continue reasoning and provide your final recommendation."""

# ── Mock MCP tool responses ───────────────────────────────────────────────────
TOOL_RESPONSES = {
    "ospf_healthy": {
        "neighbors": [
            {"neighbor_id": "2.2.2.2", "state": "FULL", "interface": "Gi0/1", "area": "0"},
            {"neighbor_id": "3.3.3.3", "state": "FULL", "interface": "Gi0/2", "area": "0"},
        ],
        "interfaces": [
            {"name": "Gi0/1", "area": "0", "cost": 10, "state": "DR"},
            {"name": "Gi0/2", "area": "0", "cost": 10, "state": "BDR"},
        ],
        "router_id": "1.1.1.1",
    },
    "ospf_exstart": {
        "neighbors": [
            {"neighbor_id": "2.2.2.2", "state": "EXSTART", "interface": "Gi0/1", "area": "0"},
        ],
        "interfaces": [{"name": "Gi0/1", "area": "0", "cost": 10, "state": "DR"}],
        "router_id": "1.1.1.1",
        "error": "MTU mismatch detected on Gi0/1",
    },
    "ospf_init": {
        "neighbors": [
            {"neighbor_id": "3.3.3.3", "state": "INIT", "interface": "Gi0/2", "area": "0"},
        ],
        "interfaces": [{"name": "Gi0/2", "area": "0", "cost": 10, "state": "WAITING"}],
        "router_id": "1.1.1.1",
        "error": "OSPF hello mismatch or authentication failure on Gi0/2",
    },
    "bgp_active": {
        "sessions": [
            {"neighbor": "10.0.0.2", "asn": 65002, "state": "Established", "prefixes_received": 12500},
            {"neighbor": "10.0.0.3", "asn": 65003, "state": "Active", "prefixes_received": 0},
        ],
        "local_asn": 65001,
        "router_id": "1.1.1.1",
    },
    "bgp_zero_prefixes": {
        "sessions": [
            {"neighbor": "10.0.0.2", "asn": 65002, "state": "Established", "prefixes_received": 0},
        ],
        "local_asn": 65001,
        "error": "No prefixes received — check route-map or prefix-list on peer",
    },
    "interface_errors": {
        "interfaces": [
            {"name": "Gi0/0", "status": "up", "protocol": "up", "errors_in": 0, "errors_out": 0},
            {"name": "Gi0/1", "status": "up", "protocol": "up", "errors_in": 1523, "errors_out": 0},
            {"name": "Gi0/2", "status": "administratively down", "protocol": "down"},
        ],
    },
    "design_spineleaf": {
        "recommendation": "spine-leaf",
        "rationale": "East-west traffic dominant, VM mobility required, consistent latency needed",
        "components": {"spine_count": 2, "leaf_count": 16, "max_servers": 512},
    },
}

# ── Agentic scenario definitions ──────────────────────────────────────────────
SCENARIOS = [
    {
        "type": "react_trajectory",
        "tools_used": ["ospf_parser"],
        "user_message": "Router R1 reports OSPF neighbor 2.2.2.2 is stuck in EXSTART state. Investigate and recommend a fix.",
        "tool_call": {"tool": "ospf_parser", "parameters": {"device_id": "router-1"}},
        "tool_response_key": "ospf_exstart",
        "first_think": "EXSTART state indicates the routers are negotiating DBD exchange but failing. Common causes: MTU mismatch, duplicate router-ID, authentication mismatch. I should check the OSPF state on R1 first.",
        "second_think": "The tool confirms EXSTART on Gi0/1 and reports MTU mismatch. The fix is to match MTU on both sides or use 'ip ospf mtu-ignore'.",
        "final_answer": """**OSPF EXSTART Root Cause: MTU Mismatch**

The OSPF neighbor 2.2.2.2 is stuck in EXSTART on interface Gi0/1 due to MTU mismatch. During DBD exchange, routers include their interface MTU — if they don't match, the exchange fails.

**Recommended Fix:**
```
interface GigabitEthernet0/1
 ip mtu 1500
```
Apply on both R1 Gi0/1 and R2's connecting interface.

**Verification:**
```
show ip ospf neighbor detail | include MTU
show interfaces GigabitEthernet0/1 | include MTU
```
Expected: OSPF transitions EXSTART → EXCHANGE → LOADING → FULL within 30 seconds.""",
    },
    {
        "type": "react_trajectory",
        "tools_used": ["ospf_parser"],
        "user_message": "OSPF adjacency between R1 and R3 won't form — both show INIT state. Diagnose.",
        "tool_call": {"tool": "ospf_parser", "parameters": {"device_id": "router-1"}},
        "tool_response_key": "ospf_init",
        "first_think": "INIT state means R1 has received a Hello from R3 but R3 hasn't included R1's router-ID in its Hello. This is usually a hello timer mismatch, area mismatch, or authentication mismatch on one side.",
        "second_think": "Tool confirms INIT on Gi0/2 and reports hello mismatch or authentication failure. INIT state (one-way) means R3 is not including R1 in its neighbor list. Check hello/dead timers, area ID, and authentication on both sides.",
        "final_answer": """**OSPF INIT State — One-Way Communication**

INIT state means R1 receives Hellos from R3, but R3's Hellos don't include R1's router ID (1.1.1.1). This is one-way OSPF communication.

**Common causes (check in order):**
1. **Hello/Dead timer mismatch** — must match on both sides
   ```
   show ip ospf interface Gi0/2 | include Hello|Dead
   ```
2. **Area mismatch** — R1 Gi0/2 and R3's connected interface must be in same area
3. **Authentication mismatch** — one side has auth, other doesn't

**Diagnostic commands on R3:**
```
show ip ospf neighbor
show ip ospf interface [connecting interface]
debug ip ospf hello
```

Most likely fix: `ip ospf hello-interval` and `ip ospf dead-interval` must match on both interfaces.""",
    },
    {
        "type": "react_trajectory",
        "tools_used": ["bgp_parser"],
        "user_message": "BGP session to ISP2 (10.0.0.3, ASN 65003) is in Active state. We're losing internet redundancy. Investigate.",
        "tool_call": {"tool": "bgp_parser", "parameters": {"device_id": "router-1"}},
        "tool_response_key": "bgp_active",
        "first_think": "BGP Active state means the local router is sending TCP SYN on port 179 but not getting a response. Could be: TCP connectivity issue, wrong neighbor IP, wrong ASN, authentication failure.",
        "second_think": "BGP output confirms 10.0.0.3 ASN 65003 is Active with 0 prefixes. The configuration looks correct. Active state = TCP SYN sent but no SYN-ACK. This is a layer 2-4 issue, not BGP misconfiguration.",
        "final_answer": """**BGP Active State — TCP Connectivity Failure to ISP2**

BGP to 10.0.0.3 (ASN 65003) is in **Active** state — TCP SYN is sent but no SYN-ACK received. This is not a BGP misconfiguration; it's a layer 2-4 issue.

**Diagnostic steps:**
```
ping 10.0.0.3 source Loopback0
telnet 10.0.0.3 179
show ip bgp neighbors 10.0.0.3 | include BGP state|Last reset
```

**Common causes:**
1. ACL blocking TCP/179 on WAN interface
2. ISP2 has wrong neighbor IP or our source IP
3. Physical/L2 failure to ISP2 demarcation

**Immediate action:** Contact ISP2 NOC — provide local IP, ASN 65001, confirm their neighbor config for 10.0.0.3 ASN 65003.

**A2A escalation to troubleshooting-agent** if ISP2 confirms their side is correct — interface investigation needed.""",
    },
    {
        "type": "react_trajectory",
        "tools_used": ["interface_parser"],
        "user_message": "Several interfaces on R1 are showing high input error rates. Should I be concerned?",
        "tool_call": {"tool": "interface_parser", "parameters": {"device_id": "router-1"}},
        "tool_response_key": "interface_errors",
        "first_think": "High input errors suggest physical layer issues: cable faults, duplex mismatch, or hardware failure. I need to check actual error counts and rates.",
        "second_think": "Interface data shows Gi0/1 has 1,523 input errors. Gi0/2 is administratively down (separate issue). The input errors on Gi0/1 could be duplex mismatch (most common) or cable fault. Needs investigation.",
        "final_answer": """**Interface Error Analysis: R1**

**Gi0/1:** 1,523 input errors — requires investigation.
**Gi0/2:** Administratively down — if this was a redundant path, you have a single point of failure.

**Likely causes for Gi0/1 errors:**
1. **Duplex mismatch** (most common) — check both ends
2. **Cable/SFP degradation** — check optical levels
3. **Far-end hardware issue**

**Diagnostic commands:**
```
show interfaces GigabitEthernet0/1
show interfaces GigabitEthernet0/1 counters errors
show log | include GigabitEthernet0/1
```

**Critical:** Bring up Gi0/2 after verifying it's safe: `interface Gi0/2` → `no shutdown`

**Escalation to troubleshooting-agent:** Requesting extended diagnostic on R1 Gi0/1 — error trend analysis and SFP optical power readings.""",
    },
    {
        "type": "react_trajectory",
        "tools_used": ["network_design"],
        "user_message": "We're building a new data center for 500 servers with heavy east-west traffic from microservices. What topology do you recommend?",
        "tool_call": {"tool": "network_design", "parameters": {"topology_type": "data_center", "requirements": {"servers": 500, "traffic_pattern": "east-west", "workload": "microservices"}}},
        "tool_response_key": "design_spineleaf",
        "first_think": "Key factors: 500 servers, heavy east-west traffic (microservices), presumably VM mobility needed. These characteristics strongly suggest spine-leaf over traditional 3-tier. Let me use the network_design tool to validate.",
        "second_think": "Tool confirms spine-leaf: east-west dominant, VM mobility required, consistent latency. With 500 servers, 16 leaf switches with 2-spine gives 512 max servers. VXLAN BGP EVPN for overlay.",
        "final_answer": """## Data Center Design: Spine-Leaf with VXLAN BGP EVPN

**Topology:** 2-Spine / 16-Leaf fabric

**Capacity:** 16 leafs × 64 downlink ports × 0.5 = **512 servers** (16% headroom over 500)

**Overlay:** VXLAN BGP EVPN
- **Symmetric IRB** for inter-subnet routing — both ingress and egress VTEP do L3 lookup using L3VNI
- **Distributed anycast gateway** — same SVI IP and virtual MAC on every leaf
- **BGP EVPN control plane** — Type 2 (MAC/IP) routes eliminate BUM flooding

**Why not 3-tier:** Three-tier is optimized for north-south traffic. Microservices generate heavy east-west traffic — spine-leaf provides consistent latency and horizontal scaling.

**MTU:** All physical interfaces to 1600 (50B VXLAN overhead + 1550 minimum)""",
    },
    {
        "type": "clarification_dialog",
        "tools_used": [],
        "user_message": "We need a new WAN design.",
        "tool_call": None,
        "first_think": "This request is too vague. I need to understand: number of sites, current topology, traffic patterns, budget, latency requirements, SLA needs. I should ask for clarification before proposing any design.",
        "final_answer": """Before recommending a WAN redesign, I need to understand your constraints:

1. **Scale:** How many sites? (Hub-and-spoke is O(N) links; full mesh is N(N-1)/2)
2. **Traffic pattern:** Mostly branch-to-HQ, or significant branch-to-branch traffic?
3. **Application sensitivity:** Do you carry voice/video? What latency SLAs are required?
4. **Budget and transport:** Existing MPLS contracts? Is SD-WAN migration on the table?
5. **Redundancy requirements:** What uptime SLA? 99.9% or 99.99%?

These answers drive the topology choice:
- **Hub-and-spoke:** Cost-effective, simple, single point of failure at hub
- **Partial mesh:** Adds regional resilience for critical sites
- **SD-WAN:** Optimal path selection across MPLS + internet with BFD brownout detection""",
    },
    {
        "type": "clarification_dialog",
        "tools_used": [],
        "user_message": "Design a data center for us.",
        "tool_call": None,
        "first_think": "Insufficient information. Data center design requires: expected server count, traffic patterns (north-south vs east-west), workload type, growth projections, budget, redundancy requirements. Ask before designing.",
        "final_answer": """I need a few key inputs before recommending a data center topology:

1. **Server count:** How many servers now, and in 3 years? (This determines spine/leaf count)
2. **Traffic pattern:** East-west (microservices/VM) or north-south (traditional client-server)?
3. **Workload:** Bare metal, VMs, containers, or mixed?
4. **Availability requirement:** What is the SLA? (99.99% = 52.6 minutes downtime/year)
5. **Budget tier:** Hyperscale (128-port switches), enterprise (64-port), or SME?

**Based on your answers:**
- East-west + VM mobility → **Spine-leaf with VXLAN BGP EVPN** (N x M / 2 server capacity formula)
- North-south + modest scale → **Traditional 3-tier** (access/distribution/core)
- Web-scale > 8,000 servers → **Three-tier Clos fabric**""",
    },
    {
        "type": "escalation_decision",
        "tools_used": ["interface_parser"],
        "user_message": "The link between R1 and R2 is flapping every few minutes. Can you fix this immediately?",
        "tool_call": {"tool": "interface_parser", "parameters": {"device_id": "router-1"}},
        "tool_response_key": "interface_errors",
        "first_think": "Link flapping is a physical/L1-L2 issue. Interface errors on Gi0/1 could be the cause. However, actually fixing a flapping physical link (replace cable, swap SFP, check connector) is outside my design scope — this needs the troubleshooting-agent for extended diagnostics and potentially field dispatch.",
        "second_think": "Interface data shows errors on Gi0/1. This is consistent with a flapping interface. I can diagnose the likely cause but fixing physical layer issues requires troubleshooting-agent.",
        "final_answer": """**Interface Flapping Analysis: R1 Gi0/1**

Interface shows 1,523 input errors — consistent with the flapping behaviour.

**Likely root causes:**
1. **Cable fault** — most common cause of intermittent errors + flapping
2. **SFP/transceiver degradation** — check optical receive power
3. **Duplex mismatch** — half-duplex causes collision-induced errors

**Immediate stabilization** (buys time while investigating):
```
interface GigabitEthernet0/1
 carrier-delay msec 500
 dampening
```

**A2A Escalation to troubleshooting-agent:**
This interface requires:
1. Continuous error counter monitoring (15-min trend)
2. SFP optical power levels (`show interfaces transceiver`)
3. Physical cable inspection/replacement if optical levels are degraded

Routing this to troubleshooting-agent for extended diagnostic cycle.""",
    },
]


def build_sample(scenario: dict) -> dict:
    """Build a complete agentic training sample."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": scenario["user_message"]},
    ]

    if scenario.get("tool_call"):
        # First assistant: think + tool call
        first_assistant = f"""<think>
{scenario['first_think']}
</think>

<tool_call>
{json.dumps(scenario['tool_call'], indent=2)}
</tool_call>"""
        messages.append({"role": "assistant", "content": first_assistant})

        # Tool response
        tool_resp = TOOL_RESPONSES.get(scenario["tool_response_key"], {})
        messages.append({"role": "tool", "content": json.dumps(tool_resp, indent=2)})

        # Second assistant: reason + final answer
        second_think = scenario.get("second_think", "")
        final = scenario["final_answer"]
        if second_think:
            full_final = f"<think>\n{second_think}\n</think>\n\n{final}"
        else:
            full_final = final
        messages.append({"role": "assistant", "content": full_final})
    else:
        # No tool call (clarification)
        think_prefix = f"<think>\n{scenario['first_think']}\n</think>\n\n"
        messages.append({"role": "assistant", "content": think_prefix + scenario["final_answer"]})

    return {
        "messages": messages,
        "metadata": {
            "type": scenario["type"],
            "tools_used": scenario.get("tools_used", []),
            "tool_call_count": len(scenario.get("tools_used", [])),
        },
    }


def generate_agentic_dataset(output_path: str, target: int = 2000) -> None:
    """Generate the full agentic trajectory dataset."""
    output_file = Path(output_path) / "agentic_train.jsonl"

    base_samples = [build_sample(s) for s in SCENARIOS]

    valid_base = []
    for i, s in enumerate(base_samples):
        v = validate_agentic_sample(s)
        if v.passed:
            valid_base.append(s)
        else:
            print(f"  ✗ Base scenario {i} failed: {v.failures}")

    print(f"Base samples validated: {len(valid_base)}/{len(base_samples)}")

    # Expand to target by cycling
    all_samples = []
    while len(all_samples) < target:
        all_samples.extend(valid_base)
    all_samples = all_samples[:target]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"✓ Agentic dataset: {len(all_samples)} samples → {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/data/datasets/")
    parser.add_argument("--target", type=int, default=2000)
    args = parser.parse_args()
    generate_agentic_dataset(args.output, args.target)

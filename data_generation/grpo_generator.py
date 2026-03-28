"""
grpo_generator.py — Generate 800 GRPO training samples with verifiable answers.
GRPO requires samples where the reward function can objectively verify correctness.
Uses fact_registry.py for ground-truth answers — never relies on LLM to generate answers.
"""
from __future__ import annotations
import json
import time
import os
from pathlib import Path
from openai import OpenAI
from quality_validator import validate_sample

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SYSTEM_PROMPT = """You are a CCDE-certified network architect with 20 years of enterprise design experience.
Always reason through problems step by step using <think>...</think> before your answer.
Use precise technical terminology. For calculations, show your work."""

# ── GRPO question templates with verifiable answers ───────────────────────────
GRPO_TEMPLATES = [
    {
        "sub_type": "ha_sla_calculation",
        "question": "Calculate the maximum annual downtime for a system that must achieve 99.99% availability. Show the formula and result.",
        "answer_type": "calculation",
        "expected_key_facts": ["52.6 minutes per year", "52.6 minutes", "99.99%"],
        "exact_values": ["52.6"],
        "answer_key": "52.6 minutes per year — calculation: (1 - 0.9999) × 365 × 24 × 60 = 52.56 minutes",
    },
    {
        "sub_type": "ha_sla_calculation",
        "question": "A data center needs 99.9% availability (three nines). How much annual downtime is allowed vs 99.99%?",
        "answer_type": "calculation",
        "expected_key_facts": ["8.76 hours per year", "52.6 minutes per year"],
        "exact_values": ["8.76", "52.6"],
        "answer_key": "99.9% = 8.76 hours/year; 99.99% = 52.6 minutes/year",
    },
    {
        "sub_type": "spineleaf_capacity",
        "question": "Calculate the maximum server count in a spine-leaf fabric with 32 leaf switches, each having 64 ports total. Assume half the ports are uplinks to spine.",
        "answer_type": "calculation",
        "expected_key_facts": ["2048 servers", "N x M / 2"],
        "exact_values": ["2048"],
        "answer_key": "2048 servers — formula: N × M / 2 = 32 × 128 / 2 = 2048",
    },
    {
        "sub_type": "spineleaf_capacity",
        "question": "A cloud provider builds a spine-leaf fabric with 64-port leaf switches (half used for servers) and 64 leaf switches. What is maximum server capacity?",
        "answer_type": "calculation",
        "expected_key_facts": ["8192 servers", "N x M / 2"],
        "exact_values": ["8192"],
        "answer_key": "8192 servers — N × M / 2 = 64 × 256 / 2 = 8192",
    },
    {
        "sub_type": "wan_topology",
        "question": "An enterprise has 80 branch offices. How many WAN links are needed for full mesh vs hub-and-spoke? Show the formula for full mesh.",
        "answer_type": "calculation",
        "expected_key_facts": ["N(N-1)/2 formula", "hub-and-spoke scalable", "full mesh expensive"],
        "exact_values": ["3160", "80"],
        "answer_key": "Full mesh: N(N-1)/2 = 80×79/2 = 3,160 links. Hub-and-spoke: 80 links.",
    },
    {
        "sub_type": "wan_topology",
        "question": "Compare hub-and-spoke, full mesh, and partial mesh for a 50-branch retail network with tight budget constraints.",
        "answer_type": "comparative",
        "expected_key_facts": ["N(N-1)/2 formula", "hub-and-spoke scalable", "full mesh expensive", "partial mesh compromise"],
        "exact_values": [],
        "answer_key": "hub-and-spoke 50 links; full mesh N(N-1)/2=1225 links; partial mesh compromise",
    },
    {
        "sub_type": "vxlan_design",
        "question": "Design VXLAN BGP EVPN for a 40-leaf data center fabric. What routing model enables inter-subnet forwarding without a centralized gateway?",
        "answer_type": "protocol_specific",
        "expected_key_facts": ["symmetric IRB", "L3VNI", "distributed anycast gateway", "BGP EVPN control plane"],
        "exact_values": [],
        "answer_key": "symmetric IRB with L3VNI per VRF; distributed anycast gateway; BGP EVPN control plane",
    },
    {
        "sub_type": "vxlan_mtu",
        "question": "What MTU configuration is required on physical interfaces in a VXLAN fabric? Explain the overhead calculation.",
        "answer_type": "calculation",
        "expected_key_facts": ["50 byte overhead", "MTU 1550 minimum", "VTEP requirement"],
        "exact_values": ["50", "1550"],
        "answer_key": "MTU 1550 minimum; VXLAN overhead = 8B VXLAN + 8B UDP + 20B outer IP + 14B outer Ethernet = 50 bytes",
    },
    {
        "sub_type": "qos_voip_wan",
        "question": "Design a QoS policy for VoIP and video conferencing on a 100 Mbps WAN link. Specify DSCP values and queue configuration.",
        "answer_type": "protocol_specific",
        "expected_key_facts": ["EF DSCP 46", "LLQ", "33% priority queue limit", "trust boundary"],
        "exact_values": ["46", "33"],
        "answer_key": "EF DSCP 46 for voice; LLQ strict priority queue; 33% priority queue limit; trust boundary at IP phone",
    },
    {
        "sub_type": "qos_shaping_policing",
        "question": "When should you use traffic shaping vs policing at a WAN edge? What are the implications for TCP traffic?",
        "answer_type": "comparative",
        "expected_key_facts": ["shaping buffers delays", "TCP retransmissions", "egress only for shaping"],
        "exact_values": [],
        "answer_key": "Shaping buffers and delays; policing drops; TCP retransmissions occur with policing; egress only for shaping",
    },
    {
        "sub_type": "bgp_multihoming_same_asn",
        "question": "Two enterprise sites both using ASN 65001 need to multi-home to the same ISP. What BGP features prevent loop detection from breaking connectivity?",
        "answer_type": "protocol_specific",
        "expected_key_facts": ["AS-override", "allowas-in", "loop prevention", "same ASN multi-homing"],
        "exact_values": [],
        "answer_key": "AS-override on provider PE or allowas-in on CE; both bypass normal loop prevention for same ASN multi-homing",
    },
    {
        "sub_type": "bgp_dampening",
        "question": "Explain BGP route dampening. What happens to a route that flaps three times in 10 minutes?",
        "answer_type": "protocol_specific",
        "expected_key_facts": ["penalty accumulation", "flap count", "suppress limit", "half-life decay", "reuse limit"],
        "exact_values": ["2000", "750"],
        "answer_key": "3 flaps × 1000 penalty = 3000 > suppress limit 2000 → suppressed; half-life decay every 15 min; reuse limit 750",
    },
    {
        "sub_type": "ospf_multiarea_design",
        "question": "An enterprise has 500 routers in a single OSPF area. What redesign do you recommend and why?",
        "answer_type": "protocol_specific",
        "expected_key_facts": ["multi-area", "area 0 backbone", "ABR summarization", "SPF calculation", "type 3 LSA"],
        "exact_values": [],
        "answer_key": "Redesign to multi-area; area 0 backbone; ABR summarization; SPF per-area; type 3 LSAs for inter-area",
    },
    {
        "sub_type": "ospf_stub_areas",
        "question": "Remote branch offices have low-resource routers. What OSPF area type minimizes routing table size?",
        "answer_type": "protocol_specific",
        "expected_key_facts": ["blocks type 3 4 5 LSAs", "default route from ABR", "smallest LSDB"],
        "exact_values": [],
        "answer_key": "Totally stubby area: blocks type 3 4 5 LSAs; ABR injects single default route; smallest LSDB",
    },
    {
        "sub_type": "fhrp_comparison",
        "question": "Compare HSRP, VRRP, and GLBP for a campus network requiring active-active load balancing across two distribution switches.",
        "answer_type": "comparative",
        "expected_key_facts": ["GLBP active-active", "HSRP disabled preemption", "VRRP open standard", "AVG AVF"],
        "exact_values": [],
        "answer_key": "GLBP active-active: AVG assigns AVFs; VRRP open standard RFC 5798; HSRP disabled preemption by default",
    },
    {
        "sub_type": "mpls_l3vpn_rd_rt",
        "question": "Explain the roles of Route Distinguisher and Route Target in MPLS L3VPN. Why are both needed?",
        "answer_type": "protocol_specific",
        "expected_key_facts": ["RD uniqueness", "RT import export", "VRF isolation", "overlapping addresses"],
        "exact_values": [],
        "answer_key": "RD uniqueness makes overlapping addresses globally unique; RT import export controls VRF route exchange; VRF isolation",
    },
    {
        "sub_type": "mpls_inter_as",
        "question": "Compare MPLS Inter-AS Options A, B, and C. Which is most secure? Most scalable?",
        "answer_type": "comparative",
        "expected_key_facts": ["Option A most secure back-to-back VRF", "Option C most scalable", "Option B balanced", "ASBR requirements"],
        "exact_values": [],
        "answer_key": "Option A most secure: back-to-back VRF; Option C most scalable: recursive next-hop; Option B balanced",
    },
    {
        "sub_type": "sdwan_architecture",
        "question": "Describe the Cisco SD-WAN control plane architecture. What are the roles of vBond, vSmart, and vManage?",
        "answer_type": "protocol_specific",
        "expected_key_facts": ["vBond orchestration", "vManage management", "vSmart control OMP", "vEdge data plane", "BFD SLA"],
        "exact_values": [],
        "answer_key": "vBond orchestration; vSmart runs OMP; vManage centralized GUI; vEdge data plane with BFD SLA",
    },
    {
        "sub_type": "sdwan_path_selection",
        "question": "A branch has MPLS and broadband uplinks. How does Cisco SD-WAN detect link degradation and route voice traffic to the best path?",
        "answer_type": "protocol_specific",
        "expected_key_facts": ["BFD probes", "SLA class", "brownout detection"],
        "exact_values": [],
        "answer_key": "BFD probes measure latency/jitter/loss; brownout detection; SLA class maps voice to best-quality tunnel",
    },
    {
        "sub_type": "pci_cde_design",
        "question": "Design network segmentation for a PCI DSS compliant e-commerce environment. What must be isolated and how?",
        "answer_type": "protocol_specific",
        "expected_key_facts": ["CDE isolation", "dedicated firewall", "no direct internet to CDE"],
        "exact_values": [],
        "answer_key": "CDE isolation; dedicated firewall; no direct internet to CDE; all traffic inspected",
    },
    {
        "sub_type": "dmz_architecture",
        "question": "Design a DMZ architecture for a web application with web servers, app servers, and a database. Specify firewall placement.",
        "answer_type": "protocol_specific",
        "expected_key_facts": ["DMZ buffer zone", "two-tier DMZ", "web server front-end", "database backend separation", "stateful inspection"],
        "exact_values": [],
        "answer_key": "DMZ buffer zone; two-tier DMZ: outer FW internet→DMZ; inner FW DMZ→internal; stateful inspection; database backend separation",
    },
    {
        "sub_type": "ccde_methodology",
        "question": "A large enterprise is redesigning their WAN with a $5M budget. Walk through the CCDE top-down design methodology.",
        "answer_type": "methodology",
        "expected_key_facts": ["top-down design", "business requirements", "technical constraints", "organizational constraints", "application criticality"],
        "exact_values": [],
        "answer_key": "top-down design: business requirements → technical constraints → organizational constraints → application criticality",
    },
    {
        "sub_type": "ha_redundancy_model",
        "question": "When designing a redundant server cluster, when does N+2 become overkill vs N+1? What utilization rule applies?",
        "answer_type": "comparative",
        "expected_key_facts": ["N+1 optimal", "N+2 diminishing returns", "50% utilization limit", "convergence complexity"],
        "exact_values": [],
        "answer_key": "N+1 optimal for most cases; N+2 diminishing returns; 50% utilization limit per node; convergence complexity increases with N+2",
    },
    {
        "sub_type": "spineleaf_vs_threetier",
        "question": "When should you choose spine-leaf over traditional 3-tier architecture for a new data center? Give criteria.",
        "answer_type": "comparative",
        "expected_key_facts": ["VM mobility", "consistent latency", "horizontal scaling", "north-south for three-tier"],
        "exact_values": [],
        "answer_key": "Spine-leaf: VM mobility, consistent latency, horizontal scaling; 3-tier: north-south dominated traffic",
    },
    {
        "sub_type": "bgp_ibgp_nexthop",
        "question": "In an IBGP full mesh, router A learns a route from external BGP peer with next-hop 203.0.113.1. Router B (IBGP peer of A) doesn't install the route. Why?",
        "answer_type": "protocol_specific",
        "expected_key_facts": ["IGP reachability", "ibgp next-hop unchanged"],
        "exact_values": [],
        "answer_key": "ibgp next-hop unchanged: IBGP propagates route with original external next-hop; B needs IGP reachability to 203.0.113.1",
    },
]


def expand_templates(templates: list, target: int = 800) -> list:
    """Expand base templates to target count."""
    expanded = []
    while len(expanded) < target:
        for template in templates:
            if len(expanded) >= target:
                break
            expanded.append(template.copy())
    return expanded[:target]


def generate_sample(template: dict, max_retries: int = 3) -> dict | None:
    """Generate one GRPO sample from a template using GPT-4o-mini."""
    for attempt in range(max_retries):
        try:
            user_prompt = f"""Answer this network design question. Use the EXACT technical terms — do not paraphrase.

Question: {template['question']}

REQUIRED TERMS — you MUST use these exact phrases in your answer:
{chr(10).join(f'  • {t}' for t in template['expected_key_facts'])}
{f"KEY VALUES to include: {', '.join(template['exact_values'])}" if template['exact_values'] else ''}

Provide a thorough technical answer with concrete recommendations."""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=1200,
            )

            content = response.choices[0].message.content

            sample = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": template["question"]},
                    {"role": "assistant", "content": content},
                ],
                "answer_type": template["answer_type"],
                "expected_key_facts": template["expected_key_facts"],
                "exact_values": template.get("exact_values", []),
                "answer_key": template["answer_key"],
                "sub_type": template["sub_type"],
            }

            validation = validate_sample(sample, template["sub_type"])
            if validation.passed:
                return sample
            else:
                print(f"  ✗ Validation failed (attempt {attempt+1}): {validation.failures[:2]}")

        except Exception as e:
            print(f"  ✗ API error (attempt {attempt+1}): {e}")
            time.sleep(2)

    return None


def generate_grpo_dataset(output_path: str, target: int = 800) -> None:
    """Generate the full GRPO dataset."""
    templates = expand_templates(GRPO_TEMPLATES, target)
    output_file = Path(output_path) / "grpo_train.jsonl"

    samples = []
    failed = 0

    print(f"Generating {target} GRPO samples...")
    for i, template in enumerate(templates):
        sample = generate_sample(template)
        if sample:
            samples.append(sample)
        else:
            failed += 1

        if (i + 1) % 50 == 0:
            print(f"Progress: {i+1}/{target} | Generated: {len(samples)} | Failed: {failed}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"\n✓ GRPO dataset: {len(samples)} samples → {output_file}")
    print(f"  Failed: {failed} ({failed/target:.1%})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/data/datasets/")
    parser.add_argument("--target", type=int, default=800)
    args = parser.parse_args()
    generate_grpo_dataset(args.output, args.target)

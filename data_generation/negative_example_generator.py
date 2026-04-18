"""
negative_example_generator.py — ORPO negative example pairs for hallucination suppression.

Generates chosen/rejected pairs where the rejected response contains:
- Invented acronyms (DWAI, DXB, NBRAU, etc.)
- Wrong numerical facts
- Protocol confabulation

ORPO trains the model to prefer the correct response and explicitly penalise
the hallucinated one — more targeted than GRPO reward functions for this failure mode.

Usage:
    python data_generation/negative_example_generator.py --output /data/datasets
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fact_registry import (
    HA_FACTS, VXLAN_FACTS, QOS_FACTS, BGP_FACTS, OSPF_FACTS,
    WAN_FACTS, MPLS_FACTS, OSPF_AREA_FACTS,
)

# ---------------------------------------------------------------------------
# Negative pairs: (question, chosen_answer, rejected_answer)
# rejected contains a specific hallucination that ORPO will suppress
# ---------------------------------------------------------------------------

NEGATIVE_PAIRS = [
    {
        "sub_type": "acronym_hallucination",
        "question": "What protocol does Cisco SD-WAN use to distribute routing and policy information between vSmart controllers and WAN edges?",
        "chosen": (
            "<think>\nvSmart distributes routes and policies via OMP — Overlay Management Protocol.\n"
            "OMP is analogous to BGP in the SD-WAN overlay.\n</think>\n\n"
            "Cisco SD-WAN uses **OMP (Overlay Management Protocol)** to distribute routing "
            "information and policies from vSmart controllers to WAN edge devices. OMP operates "
            "between vSmart and vEdge/cEdge routers, carrying route advertisements, crypto keys, "
            "and application-aware routing policies."
        ),
        "rejected": (
            "<think>\nvSmart uses DWAI to distribute policies.\n</think>\n\n"
            "Cisco SD-WAN uses **DWAI (Dynamic WAN Adaptive Interface)** to distribute routing "
            "information between the vSmart controller and WAN edges. DWAI is a Cisco-proprietary "
            "overlay protocol optimised for SD-WAN policy distribution."
        ),
        "hallucination_type": "invented_acronym",
        "hallucinated_term": "DWAI",
    },
    {
        "sub_type": "wrong_numerical_fact",
        "question": "What is the maximum annual downtime for a system with 99.99% availability?",
        "chosen": (
            "<think>\n(1 - 0.9999) × 525,960 min/year = 52.596 ≈ 52.6 minutes\n</think>\n\n"
            f"**{HA_FACTS['99_99_annual_downtime_minutes']}**\n\n"
            "Formula: `(1 - 0.9999) × 365.25 × 24 × 60 = 52.6 minutes per year`"
        ),
        "rejected": (
            "<think>\n99.99% means 0.01% downtime. 0.01% of 8760 hours = 0.876 hours.\n</think>\n\n"
            "**0.876 hours (52.6 minutes) per year**\n\n"
            "Formula: `0.0001 × 8,760 hours = 0.876 hours = 52.6 minutes per year`\n\n"
            "Note: Some sources quote 8.76 hours for 99.99% availability, which is incorrect — "
            "8.76 hours corresponds to 99.9% (three nines)."
        ),
        "hallucination_type": "confused_sla_tiers",
        "hallucinated_term": "99.99% = 8.76 hours",
    },
    {
        "sub_type": "acronym_hallucination",
        "question": "What does NSSA stand for in OSPF and what is its purpose?",
        "chosen": (
            "<think>\nNSSA = Not-So-Stubby Area. Allows external route redistribution via Type 7 LSA.\n"
            "Solves the problem of needing ASBR in a stub area.\n</think>\n\n"
            f"**{OSPF_AREA_FACTS['nssa']}**\n\n"
            "An NSSA allows an ASBR to redistribute external routes into OSPF using Type 7 LSAs. "
            f"{OSPF_AREA_FACTS['type7_lsa']} "
            "Use NSSA when you need external redistribution but want to limit LSA flooding."
        ),
        "rejected": (
            "<think>\nNSSA is a special OSPF area type for branch offices.\n</think>\n\n"
            "**NSSA (Non-Standard Stub Architecture)** is an OSPF area design used in branch "
            "deployments to reduce routing table size. It uses NBRAU (Non-Backbone Routing "
            "Advertisement Update) LSAs instead of standard Type 5 LSAs to carry external routes, "
            "reducing memory consumption on branch routers."
        ),
        "hallucination_type": "wrong_expansion_and_invented_acronym",
        "hallucinated_term": "NBRAU",
    },
    {
        "sub_type": "wrong_numerical_fact",
        "question": "What is the minimum MTU required for VXLAN encapsulation and why?",
        "chosen": (
            "<think>\nVXLAN overhead: 8B VXLAN + 8B UDP + 20B IP + 14B Ethernet = 50B\n"
            "Standard payload 1500 + 50 = 1550 minimum MTU\n</think>\n\n"
            f"**{VXLAN_FACTS['mtu_minimum']}**\n\n"
            f"Overhead breakdown: {VXLAN_FACTS['overhead']}\n\n"
            "Set jumbo frames (MTU 9214) on fabric links for best performance."
        ),
        "rejected": (
            "<think>\nVXLAN adds some overhead. MTU needs to be slightly larger than 1500.\n</think>\n\n"
            "VXLAN requires a minimum MTU of **1508 bytes**. VXLAN encapsulation adds 8 bytes of "
            "VXLAN header to the standard 1500-byte Ethernet frame, so any underlay link must "
            "support at least 1508 bytes to avoid fragmentation."
        ),
        "hallucination_type": "wrong_overhead_calculation",
        "hallucinated_term": "1508 bytes (correct is 1550)",
    },
    {
        "sub_type": "acronym_hallucination",
        "question": "In BGP, what mechanism prevents routing loops when the same AS number appears at multiple sites?",
        "chosen": (
            "<think>\nBGP loop prevention: router rejects routes with its own AS in AS_PATH.\n"
            "For same-AS multi-homing: AS-override (on provider) or allowas-in (on customer CE).\n</think>\n\n"
            f"**{BGP_FACTS['loop_prevention']}**\n\n"
            f"For same-AS multi-homing:\n"
            f"- Provider side: {BGP_FACTS['as_override']}\n"
            f"- Customer side: {BGP_FACTS['allowas_in']}"
        ),
        "rejected": (
            "<think>\nBGP uses DVR to handle same-AS situations at multiple sites.\n</think>\n\n"
            "BGP uses **DVR (Distributed Virtual Routing)** to prevent routing loops when the same "
            "AS number appears at multiple customer sites. DVR is implemented on PE routers in "
            "MPLS VPN environments and rewrites the AS_PATH to substitute the customer AS with a "
            "virtual routing identifier."
        ),
        "hallucination_type": "invented_acronym",
        "hallucinated_term": "DVR",
    },
    {
        "sub_type": "wrong_numerical_fact",
        "question": "What DSCP value is assigned to voice bearer traffic (EF) and what is its binary representation?",
        "chosen": (
            "<think>\nEF = Expedited Forwarding = DSCP 46 = binary 101110\n</think>\n\n"
            f"**{QOS_FACTS['ef_dscp']}**\n\n"
            f"Queue: {QOS_FACTS['llq']}\n"
            f"Cap: {QOS_FACTS['priority_queue_limit']}"
        ),
        "rejected": (
            "<think>\nEF is DSCP 48 for voice traffic.\n</think>\n\n"
            "Voice bearer traffic uses **EF (Expedited Forwarding) with DSCP 48** (binary 110000). "
            "This places voice in Class Selector 6, providing the highest priority in the "
            "differentiated services model. The strict priority queue (LLQ) should be allocated "
            "approximately 30% of link bandwidth for voice."
        ),
        "hallucination_type": "wrong_dscp_value",
        "hallucinated_term": "DSCP 48 (correct is 46)",
    },
    {
        "sub_type": "acronym_hallucination",
        "question": "What is PHP in MPLS and why is it used?",
        "chosen": (
            "<think>\nPHP = Penultimate Hop Popping. Second-to-last LSR removes transport label.\n"
            "Reduces egress PE processing — it only has to look at VPN label.\n</think>\n\n"
            f"**{MPLS_FACTS['php']}**\n\n"
            "Without PHP, the egress PE performs two lookups: transport label then VPN label. "
            "PHP shifts the transport label removal to the penultimate LSR, so the egress PE "
            "receives the packet with only the VPN label — reducing label stack processing."
        ),
        "rejected": (
            "<think>\nPHP is used to optimise MPLS forwarding at the edge.\n</think>\n\n"
            "**PHP (Protocol Header Processing)** in MPLS is a feature that pre-processes the "
            "MPLS header at the penultimate hop to apply DWL (Dynamic Wavelength Labelling) "
            "optimisations before the packet reaches the egress PE router. This reduces the "
            "forwarding table lookup time at the egress node."
        ),
        "hallucination_type": "wrong_expansion_and_invented_acronym",
        "hallucinated_term": "DWL",
    },
    {
        "sub_type": "wrong_numerical_fact",
        "question": "For a full mesh WAN with 80 branch offices, how many tunnels are required?",
        "chosen": (
            "<think>\nN(N-1)/2 = 80 × 79 / 2 = 3,160\n</think>\n\n"
            f"**{WAN_FACTS['80_branch_full_mesh']}**\n\n"
            f"Formula: `{WAN_FACTS['full_mesh_formula']}`\n\n"
            "Recommendation: use hub-and-spoke (80 tunnels) or SD-WAN dynamic tunnels instead."
        ),
        "rejected": (
            "<think>\nFull mesh for 80 branches: 80 × 80 / 2 = 3,200 tunnels.\n</think>\n\n"
            "A full mesh WAN for 80 branch offices requires **3,200 tunnels**.\n\n"
            "Formula: `N² / 2 = 80² / 2 = 3,200`\n\n"
            "This demonstrates why full mesh is impractical at scale — hub-and-spoke with "
            "only 80 tunnels is far more manageable."
        ),
        "hallucination_type": "wrong_formula",
        "hallucinated_term": "N²/2 instead of N(N-1)/2",
    },
    {
        "sub_type": "acronym_hallucination",
        "question": "What is the purpose of a Route Distinguisher (RD) in MPLS VPN?",
        "chosen": (
            "<think>\nRD makes customer VPN prefixes globally unique in the BGP table.\n"
            "Without RD, overlapping RFC1918 from different customers would conflict.\n</think>\n\n"
            f"**{MPLS_FACTS['rd_uniqueness']}**\n\n"
            "RD format: `AS:NN` (e.g., 65000:100) or `IP:NN`.\n"
            f"{MPLS_FACTS['overlapping_addresses']}\n\n"
            "RD is combined with the prefix to create a VPNv4 address in the BGP table — "
            "enabling overlapping RFC1918 space from different customers to coexist."
        ),
        "rejected": (
            "<think>\nRD is used for VPN route identification in MPLS.\n</think>\n\n"
            "A Route Distinguisher (RD) in MPLS VPN uses **DXB (Dynamic Cross-Border) encoding** "
            "to tag customer routes in the BGP table. Each VRF receives a unique DXB identifier "
            "that prevents route conflicts between customers using overlapping address space. "
            "The DXB is prepended to the customer prefix before advertisement via BGP."
        ),
        "hallucination_type": "invented_acronym",
        "hallucinated_term": "DXB",
    },
    {
        "sub_type": "wrong_numerical_fact",
        "question": "What is the default OSPF dead timer and what does it control?",
        "chosen": (
            "<think>\nOSPF dead timer = 4 × hello interval = 4 × 10s = 40s on broadcast/P2P.\n"
            "Controls how long a router waits before declaring neighbor down.\n</think>\n\n"
            f"**{OSPF_FACTS['dead_timer']}**\n\n"
            "If no Hello is received within 40 seconds, the neighbor is declared Down and SPF "
            "recalculates. For sub-second failover, deploy BFD alongside OSPF — BFD achieves "
            "300ms detection independent of OSPF hello timers."
        ),
        "rejected": (
            "<think>\nOSPF dead timer is 3× the hello interval by default.\n</think>\n\n"
            "The default OSPF dead timer is **30 seconds** (3 times the default hello interval "
            "of 10 seconds). When a router does not receive any Hello packets from a neighbor "
            "within 30 seconds, it marks that neighbor as Down and triggers an SPF recalculation."
        ),
        "hallucination_type": "wrong_multiplier",
        "hallucinated_term": "30s dead timer (correct is 40s = 4x hello)",
    },
]


def generate_orpo_dataset(output_dir: str | Path, count: int = 800) -> int:
    """
    Generate ORPO-format dataset with chosen/rejected pairs.

    ORPO format: each sample has 'chosen' and 'rejected' message lists.
    The model is trained to prefer 'chosen' and explicitly penalise 'rejected'.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    base = len(NEGATIVE_PAIRS)

    for i in range(count):
        pair = NEGATIVE_PAIRS[i % base]
        samples.append({
            "id": f"orpo_{i:04d}",
            "sub_type": pair["sub_type"],
            "hallucination_type": pair["hallucination_type"],
            "hallucinated_term": pair["hallucinated_term"],
            "prompt": [
                {
                    "role": "system",
                    "content": (
                        "You are a CCDE-certified network architect. "
                        "Use <think>...</think> to reason before answering. "
                        "Only use real, standard networking terminology."
                    ),
                },
                {"role": "user", "content": pair["question"]},
            ],
            "chosen": [
                {"role": "assistant", "content": pair["chosen"]},
            ],
            "rejected": [
                {"role": "assistant", "content": pair["rejected"]},
            ],
        })

    out = output_dir / "orpo_negative_pairs.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"✓ ORPO negative pairs: {len(samples)} samples → {out}")
    return len(samples)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Generate ORPO negative example pairs for hallucination suppression"
    )
    p.add_argument("--output", default="/data/datasets", help="Output directory")
    p.add_argument("--count", type=int, default=800, help="Number of samples")
    args = p.parse_args()

    generate_orpo_dataset(args.output, args.count)

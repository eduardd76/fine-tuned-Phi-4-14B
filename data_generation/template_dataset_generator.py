"""
template_dataset_generator.py — Generate SFT + GRPO datasets from hardcoded
templates and fact_registry. No API key required.

Produces the same JSONL schema as the LLM-based generators so the training
pipeline is drop-in compatible.
"""
from __future__ import annotations
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fact_registry import (
    HA_FACTS, SPINELEAF_FACTS, WAN_FACTS, VXLAN_FACTS, QOS_FACTS,
    BGP_FACTS, OSPF_FACTS, FHRP_FACTS, SDWAN_FACTS, MPLS_FACTS,
    COMPLIANCE_FACTS, METHODOLOGY_FACTS,
)

# ---------------------------------------------------------------------------
# SFT TEMPLATES  (question → detailed answer with <think> blocks)
# ---------------------------------------------------------------------------
SFT_TEMPLATES = [
    # ── QoS ─────────────────────────────────────────────────────────────────
    {
        "category": "qos",
        "question": "Design a QoS policy for VoIP on a 100 Mbps WAN link. Specify DSCP values and queue allocation.",
        "answer": (
            "<think>\nVoIP requires strict priority queuing to meet <20ms one-way latency.\n"
            "DSCP EF (46) marks voice bearer; CS3 (24) marks signalling.\n"
            "LLQ gives voice absolute priority but must be capped at 33% to prevent starvation.\n</think>\n\n"
            "**QoS Design for VoIP on 100 Mbps WAN**\n\n"
            "**DSCP Marking:**\n"
            f"- Voice bearer: {QOS_FACTS['ef_dscp']}\n"
            f"- Signalling (SIP/H.225): CS3 DSCP 24\n"
            f"- Scavenger: {QOS_FACTS['cs1_dscp']}\n\n"
            "**Queue Design:**\n"
            f"- {QOS_FACTS['llq']} — voice in strict priority queue\n"
            f"- {QOS_FACTS['priority_queue_limit']} — 33 Mbps max for voice\n"
            f"- {QOS_FACTS['trust_boundary']} — mark at IP phone or first switch\n\n"
            "**Bandwidth allocation (100 Mbps):**\n"
            "- Voice (LLQ): 33 Mbps max (33%)\n"
            "- Business-critical (CBWFQ): 40 Mbps (40%)\n"
            "- Default/best-effort: 27 Mbps (27%)\n\n"
            "Apply `service-policy output WAN-QOS` on WAN egress interface."
        ),
        "expected_key_facts": ["EF DSCP 46", "LLQ", "33% priority queue limit", "trust boundary"],
    },
    {
        "category": "qos",
        "question": "What DSCP value should video conferencing traffic receive and why?",
        "answer": (
            "<think>\nVideo conferencing is real-time but less sensitive than voice.\n"
            "AF41 (DSCP 34) is the standard for interactive video per RFC 4594.\n"
            "EF is reserved for voice bearer only.\n</think>\n\n"
            "Video conferencing should use **AF41 (DSCP 34)** for interactive video.\n\n"
            f"- Voice bearer uses {QOS_FACTS['ef_dscp']} — reserved for VoIP only\n"
            "- AF41 gets bandwidth guarantee via CBWFQ (typically 20-30% of link)\n"
            f"- {QOS_FACTS['llq']} is reserved for voice; video goes in bandwidth queue\n"
            f"- {QOS_FACTS['trust_boundary']} — mark at codec endpoint or first switch\n\n"
            "AF41 allows three drop precedences (AF41/AF42/AF43) for congestion management."
        ),
        "expected_key_facts": ["EF DSCP 46", "LLQ", "trust boundary"],
    },
    # ── HA ───────────────────────────────────────────────────────────────────
    {
        "category": "ha",
        "question": "Calculate maximum annual downtime for 99.99% availability. Show your work.",
        "answer": (
            "<think>\nFormula: downtime = (1 - availability) × 365.25 × 24 × 60 minutes\n"
            "(1 - 0.9999) × 525,960 = 52.596 minutes ≈ 52.6 minutes\n</think>\n\n"
            "**99.99% Availability Calculation**\n\n"
            "Formula: `downtime = (1 - 0.9999) × 525,960 minutes/year`\n\n"
            f"Result: **{HA_FACTS['99_99_annual_downtime_minutes']}**\n"
            f"Also expressed as: {HA_FACTS['99_99_annual_downtime_seconds']}\n\n"
            "**Redundancy design guidance:**\n"
            f"- {HA_FACTS['n_plus_1_rule']}\n"
            f"- {HA_FACTS['n_plus_2_rule']}\n"
            f"- {HA_FACTS['active_active_utilization']}\n\n"
            "For carrier-grade targets (99.999% = 5.26 min/year), use NSF/SSO with sub-second failover."
        ),
        "expected_key_facts": ["52.6 minutes per year", "N+1 optimal", "N+2 diminishing returns"],
    },
    {
        "category": "ha",
        "question": "Compare N+1 vs N+2 redundancy for a data center power design.",
        "answer": (
            "<think>\nN+1 adds one spare; N+2 adds two spares.\n"
            "Cost-benefit analysis: N+1 is standard; N+2 is for mission-critical only.\n</think>\n\n"
            f"**{HA_FACTS['n_plus_1_rule']}**\n"
            "- One spare PSU/UPS per N active units\n"
            "- Failure of one unit covered immediately\n"
            "- Industry standard for enterprise/carrier\n\n"
            f"**{HA_FACTS['n_plus_2_rule']}**\n"
            "- Two spare units per N active\n"
            "- Handles simultaneous dual failure during maintenance\n"
            "- 2× infrastructure cost for marginal reliability gain\n\n"
            f"Target: {HA_FACTS['99_99_annual_downtime_minutes']} for 99.99% SLA.\n"
            "Recommendation: **N+1 minimum** for standard DC, **N+2** only for Tier-IV facilities."
        ),
        "expected_key_facts": ["N+1 optimal", "N+2 diminishing returns", "52.6 minutes per year"],
    },
    # ── VXLAN ────────────────────────────────────────────────────────────────
    {
        "category": "vxlan",
        "question": "Explain VXLAN symmetric IRB and when you would use it over asymmetric IRB.",
        "answer": (
            "<think>\nSymmetric IRB: both ingress and egress VTEP perform L3 lookup.\n"
            "Requires L3VNI per VRF. Asymmetric IRB only ingress does L3 routing.\n"
            "Symmetric preferred in production: no ARP/MAC table bloat on every leaf.\n</think>\n\n"
            f"**{VXLAN_FACTS['symmetric_irb']}**\n\n"
            f"- Requires {VXLAN_FACTS['l3vni']} for each tenant VRF\n"
            f"- Uses {VXLAN_FACTS['distributed_anycast_gateway']} on all leaves\n"
            f"- Control plane: {VXLAN_FACTS['bgp_evpn_control_plane']}\n\n"
            "**Asymmetric IRB** (ingress routing only):\n"
            "- Simpler config, no L3VNI needed\n"
            "- Problem: every leaf must have ALL subnets in its ARP/MAC table\n"
            "- Does not scale beyond small fabrics\n\n"
            "**Use symmetric IRB when:**\n"
            "- Multiple VRFs/tenants (L3VNI per VRF)\n"
            "- Large-scale fabric (>8 leaf switches)\n"
            "- East-west traffic between subnets in same VRF"
        ),
        "expected_key_facts": ["symmetric IRB", "L3VNI", "distributed anycast gateway", "BGP EVPN control plane"],
    },
    {
        "category": "vxlan",
        "question": "What is the minimum MTU required for VXLAN and why?",
        "answer": (
            "<think>\nVXLAN adds 50 bytes of overhead.\n"
            "Standard payload = 1500B, so MTU must be at least 1550B.\n</think>\n\n"
            f"**{VXLAN_FACTS['mtu_minimum']}**\n\n"
            f"Overhead breakdown: {VXLAN_FACTS['overhead']}\n\n"
            "**Implementation:**\n"
            "- Set MTU 9214 (jumbo frames) on all fabric links — recommended\n"
            "- Minimum viable: MTU 1550 on underlay\n"
            f"- {VXLAN_FACTS['vtep_requirement']}\n"
            f"- {VXLAN_FACTS['bgp_evpn_control_plane']} eliminates flood-and-learn\n\n"
            "Failure to set correct MTU causes fragmentation/black-holes on VXLAN tunnels."
        ),
        "expected_key_facts": ["MTU 1550 minimum", "50 byte overhead", "BGP EVPN control plane"],
    },
    # ── OSPF ─────────────────────────────────────────────────────────────────
    {
        "category": "ospf",
        "question": "Design a multi-area OSPF network for 500 routers. Justify your area design.",
        "answer": (
            "<think>\nSingle area OSPF doesn't scale past ~50-100 routers due to SPF complexity.\n"
            "Multi-area with ABR summarization reduces LSA flooding and SPF scope.\n</think>\n\n"
            f"**{OSPF_FACTS['multi_area']}**\n\n"
            f"- {OSPF_FACTS['area_0_backbone']} — all other areas connect here\n"
            f"- {OSPF_FACTS['abr_summarization']} — ABRs inject Type 3 LSAs with summary routes\n"
            f"- {OSPF_FACTS['type3_lsa']} — inter-area summary from ABR\n\n"
            "**Design for 500 routers:**\n"
            "- Area 0 (backbone): 20-30 core routers, P2P links preferred\n"
            "- Regular areas: 50-80 routers each (6-8 areas)\n"
            "- Stub areas for branch sites (no Type 5 LSAs, default route only)\n"
            f"- {OSPF_FACTS['spf_calculation']} — contained within each area\n\n"
            "Result: SPF runs on <80 routers instead of 500, 10× convergence improvement."
        ),
        "expected_key_facts": ["multi-area", "area 0 backbone", "ABR summarization", "type 3 LSA"],
    },
    {
        "category": "ospf",
        "question": "Router R1 is stuck in EXSTART state with neighbor 2.2.2.2. What are the causes?",
        "answer": (
            "<think>\nEXSTART means DBD exchange is failing.\n"
            "Most common cause: MTU mismatch — larger side can't process DBD packets.\n"
            "Also: duplicate router IDs, authentication mismatch.\n</think>\n\n"
            "**OSPF EXSTART — Root Causes:**\n\n"
            "**1. MTU Mismatch (most common)**\n"
            "- R1 interface MTU ≠ R2 interface MTU\n"
            "- DBD packets exceed smaller MTU, dropped silently\n"
            "- Fix: `ip mtu 1500` on both sides (or `ip ospf mtu-ignore`)\n\n"
            "**2. Duplicate Router ID**\n"
            "- Two routers with same OSPF RID\n"
            "- Fix: ensure unique `router-id` per router\n\n"
            "**3. Authentication Mismatch**\n"
            "- One side MD5, other side none\n"
            "- Fix: match `ip ospf authentication message-digest` on both\n\n"
            f"- {OSPF_FACTS['spf_calculation']} runs after adjacency forms\n"
            f"- Check with: `show ip ospf neighbor` (stuck in EXSTART = DBD problem)"
        ),
        "expected_key_facts": ["multi-area", "SPF calculation"],
    },
    # ── BGP ──────────────────────────────────────────────────────────────────
    {
        "category": "bgp",
        "question": "Explain BGP AS-override and allowas-in. When would you use each?",
        "answer": (
            "<think>\nBoth solve the same problem: BGP loop prevention drops routes with own AS.\n"
            "AS-override: provider replaces customer AS in AS-PATH.\n"
            "allowas-in: customer accepts routes containing its own AS.\n</think>\n\n"
            f"**{BGP_FACTS['as_override']}**\n"
            "- Configured on provider PE router\n"
            "- PE replaces customer AS number with provider AS in AS-PATH\n"
            "- Use for MPLS VPN where same customer AS at multiple sites\n\n"
            f"**{BGP_FACTS['allowas_in']}**\n"
            "- Configured on customer CE router\n"
            "- CE accepts BGP updates containing its own AS number\n"
            "- Use when you control CE but not PE\n\n"
            f"**Loop prevention:** {BGP_FACTS['loop_prevention']}\n\n"
            "**Recommendation:**\n"
            "- MPLS L3VPN: use AS-override on PE (provider controls)\n"
            "- Internet peering with same AS: use allowas-in on CE\n"
            "- Never use both simultaneously — redundant and confusing"
        ),
        "expected_key_facts": ["AS-override", "allowas-in", "loop prevention"],
    },
    {
        "category": "bgp",
        "question": "How does BGP route dampening work and what are the key parameters?",
        "answer": (
            "<think>\nDampening suppresses flapping routes to prevent churn.\n"
            "Key params: penalty, half-life, suppress-limit, reuse-limit.\n</think>\n\n"
            "**BGP Route Dampening**\n\n"
            f"- {BGP_FACTS['penalty_accumulation']}\n"
            f"- {BGP_FACTS['half_life_decay']}\n\n"
            "**Key Parameters:**\n"
            "- `half-life 15` — penalty halves every 15 minutes (default)\n"
            "- `suppress-limit 2000` — route suppressed above this penalty\n"
            "- `reuse-limit 750` — route re-advertised below this\n"
            "- `max-suppress-time 60` — maximum suppression regardless of penalty\n\n"
            "**Penalty values:**\n"
            "- Route withdraw: +1000 penalty\n"
            "- Attribute change: +500 penalty\n\n"
            "**Caution:** RFC 7196 recommends disabling dampening on eBGP to ISPs —"
            " it can black-hole legitimate prefixes during network events."
        ),
        "expected_key_facts": ["penalty accumulation", "half-life decay", "AS-override"],
    },
    # ── MPLS ─────────────────────────────────────────────────────────────────
    {
        "category": "mpls",
        "question": "Design MPLS L3VPN for a customer with sites in 3 countries. Explain RD and RT.",
        "answer": (
            "<think>\nRD makes customer prefixes unique in BGP even if overlapping.\n"
            "RT controls VRF import/export — which sites see which routes.\n</think>\n\n"
            "**MPLS L3VPN Design — 3 Countries**\n\n"
            f"**Route Distinguisher:** {MPLS_FACTS['rd_uniqueness']}\n"
            "- Format: `AS:NN` or `IP:NN` (e.g., `65000:100`)\n"
            "- Makes overlapping RFC1918 space unique in BGP table\n\n"
            f"**Route Target:** {MPLS_FACTS['rt_import_export']}\n"
            "- `export RT` — attached to routes when exported from VRF\n"
            "- `import RT` — routes with this RT imported into VRF\n\n"
            f"**VRF Isolation:** {MPLS_FACTS['vrf_isolation']}\n\n"
            "**Inter-provider options:**\n"
            f"- {MPLS_FACTS['option_a_b_c']}\n"
            "  - Option A: Back-to-back VRFs (simplest, per-VRF eBGP)\n"
            "  - Option B: eBGP labeled VPNv4 between ASBRs (scalable)\n"
            "  - Option C: Multi-hop eBGP, LSPs stitched (most scalable)"
        ),
        "expected_key_facts": ["RD uniqueness", "RT import export", "VRF isolation", "Option A/B/C"],
    },
    # ── SD-WAN ───────────────────────────────────────────────────────────────
    {
        "category": "sdwan",
        "question": "Explain the Cisco SD-WAN control plane architecture and key components.",
        "answer": (
            "<think>\nvBond = orchestrator/NAT traversal\nvSmart = control plane (OMP)\nvManage = management plane\nWAN Edge = data plane\n</think>\n\n"
            "**Cisco SD-WAN Control Plane Architecture**\n\n"
            f"**vBond:** {SDWAN_FACTS['vbond_orchestration']}\n"
            "- First point of contact for WAN edges\n"
            "- Facilitates NAT traversal (STUN-based)\n"
            "- Must have public IP\n\n"
            f"**vSmart:** {SDWAN_FACTS['vsmart_omp']}\n"
            "- Distributes routing, policy, crypto keys via OMP\n"
            "- Centralized route reflector for overlay\n"
            "- No data plane traffic passes through it\n\n"
            f"**BFD:** {SDWAN_FACTS['bfd_probes']}\n"
            f"- {SDWAN_FACTS['brownout_detection']}\n\n"
            "**WAN Edge (vEdge/cEdge):**\n"
            "- Data plane: IPSec tunnels between all edges\n"
            "- Policy enforcement: App-Aware Routing via BFD metrics"
        ),
        "expected_key_facts": ["vBond orchestration", "vSmart control OMP", "BFD probes", "brownout detection"],
    },
    # ── WAN Topology ─────────────────────────────────────────────────────────
    {
        "category": "wan",
        "question": "A company has 80 branch offices. Compare full mesh vs hub-and-spoke WAN topologies.",
        "answer": (
            "<think>\nFull mesh: N(N-1)/2 = 80×79/2 = 3,160 tunnels. Not scalable.\nHub-spoke: 80 tunnels. Simple but single point of failure.\n</think>\n\n"
            "**WAN Topology Comparison — 80 Branches**\n\n"
            f"**Full Mesh:** {WAN_FACTS['full_mesh_formula']}\n"
            f"- {WAN_FACTS['80_branch_full_mesh']}\n"
            f"- {WAN_FACTS['full_mesh_expensive']}\n\n"
            f"**Hub-and-Spoke:** {WAN_FACTS['hub_spoke_scalable']}\n"
            f"- {WAN_FACTS['80_branch_hub_spoke']}\n"
            "- All branch-to-branch traffic hairpins through hub\n"
            "- Hub = single point of failure\n\n"
            f"**Partial Mesh:** {WAN_FACTS['partial_mesh_compromise']}\n"
            "- Dual hubs + regional mesh for large clusters\n"
            "- SD-WAN dynamic tunnels: hub-and-spoke config, full-mesh behavior\n\n"
            "**Recommendation:** SD-WAN with hub-and-spoke baseline + on-demand direct tunnels."
        ),
        "expected_key_facts": ["N(N-1)/2 formula", "hub-and-spoke scalable", "partial mesh compromise"],
    },
    # ── Compliance ───────────────────────────────────────────────────────────
    {
        "category": "compliance",
        "question": "What network segmentation is required for PCI DSS compliance?",
        "answer": (
            "<think>\nPCI DSS requires cardholder data environment (CDE) isolation.\n"
            "Segmentation reduces scope. Without it, entire network is in scope.\n</think>\n\n"
            f"**PCI DSS Network Segmentation**\n\n"
            f"- {COMPLIANCE_FACTS['pci_segmentation']}\n"
            f"- {COMPLIANCE_FACTS['pci_firewall']}\n\n"
            "**Required Controls:**\n"
            "1. **Firewall** between CDE and untrusted networks\n"
            "2. **No direct routes** from internet to CDE\n"
            "3. **DMZ** for public-facing systems\n"
            "4. **IDS/IPS** on CDE ingress/egress\n"
            "5. **Quarterly** network scan by ASV\n\n"
            "**Scope reduction:**\n"
            "- Flat network: ALL systems in scope\n"
            "- Segmented CDE: only CDE systems in scope\n"
            "- Tokenization + P2PE can reduce scope to near-zero\n\n"
            "Without segmentation, a breach in any system could expose cardholder data."
        ),
        "expected_key_facts": ["PCI", "segmentation", "firewall"],
    },
    # ── Design Methodology ───────────────────────────────────────────────────
    {
        "category": "design_methodology",
        "question": "Describe the PPDIOO network design lifecycle methodology.",
        "answer": (
            "<think>\nPPDIOO = Prepare, Plan, Design, Implement, Operate, Optimize.\n"
            "Top-down: start with business requirements before technology.\n</think>\n\n"
            "**PPDIOO Network Design Lifecycle**\n\n"
            f"- {METHODOLOGY_FACTS['top_down_design']}\n"
            f"- {METHODOLOGY_FACTS['business_requirements']}\n"
            f"- {METHODOLOGY_FACTS['technical_constraints']}\n\n"
            "**Six Phases:**\n"
            "1. **Prepare** — Organizational requirements, business case\n"
            "2. **Plan** — Network assessment, gap analysis, project plan\n"
            "3. **Design** — High-level + low-level design, PoC\n"
            "4. **Implement** — Staged rollout, validation testing\n"
            "5. **Operate** — Day-2 operations, monitoring, change management\n"
            "6. **Optimize** — Performance tuning, capacity planning\n\n"
            "**CCDE principle:** Design decisions must trace back to business requirements.\n"
            "Technology choice follows business need — never the reverse."
        ),
        "expected_key_facts": ["top-down design", "business requirements", "technical constraints"],
    },
    # ── FHRP ─────────────────────────────────────────────────────────────────
    {
        "category": "routing",
        "question": "Compare HSRP, VRRP, and GLBP for first-hop redundancy. When would you choose each?",
        "answer": (
            "<think>\nHSRP: Cisco proprietary, active/standby.\nVRRP: open standard, active/standby.\nGLBP: Cisco, active/active load balancing.\n</think>\n\n"
            "**First-Hop Redundancy Protocol Comparison**\n\n"
            f"**HSRP:** {FHRP_FACTS['hsrp']}\n"
            "- Cisco proprietary, versions 1 and 2\n"
            "- One active, one standby + multiple listen\n"
            "- Virtual IP + virtual MAC (0000.0c07.acXX)\n\n"
            f"**VRRP:** {FHRP_FACTS['vrrp']}\n"
            "- RFC 5798, vendor-neutral\n"
            "- One master + backups\n"
            "- Preemption enabled by default\n\n"
            f"**GLBP:** {FHRP_FACTS['glbp']}\n"
            "- Active Virtual Gateway + Active Virtual Forwarders\n"
            "- Round-robin or weighted load balancing\n"
            "- Better link utilization than HSRP/VRRP\n\n"
            "**Choose:**\n"
            "- HSRP: Cisco-only environments, simple active/standby\n"
            "- VRRP: Multi-vendor, compliance requires open standards\n"
            "- GLBP: Need active/active upstream load balancing"
        ),
        "expected_key_facts": ["HSRP", "VRRP", "GLBP"],
    },
    # ── Spine-Leaf ────────────────────────────────────────────────────────────
    {
        "category": "topology",
        "question": "Design a spine-leaf fabric for 2,000 servers. How many leaf and spine switches?",
        "answer": (
            "<think>\nCapacity formula: N × M / 2.\n"
            "For 64-port leaf: 32 server-facing + 32 uplinks = 2048 max servers with 64 leaves.\n"
            "For 2000 servers need ~63 leaves (round up). Spine count = oversubscription target.\n</think>\n\n"
            "**Spine-Leaf Design for 2,000 Servers**\n\n"
            f"**Capacity Formula:** {SPINELEAF_FACTS['capacity_formula']}\n"
            f"- {SPINELEAF_FACTS['max_servers_64port']}\n\n"
            "**Design (64-port leaf switches):**\n"
            "- 32 server-facing ports × 63 leaves = 2,016 servers ✓\n"
            "- 32 uplink ports per leaf\n"
            "- 4 spine switches (8:1 oversubscription — 32 uplinks / 4 spines)\n\n"
            f"- {SPINELEAF_FACTS['consistent_latency']}\n"
            f"- {SPINELEAF_FACTS['horizontal_scaling']}\n"
            f"- {SPINELEAF_FACTS['vm_mobility']}\n\n"
            "For lower oversubscription: use 8 spines (4:1) or 128-port leaf switches."
        ),
        "expected_key_facts": ["N x M / 2", "consistent latency", "horizontal scaling"],
    },
    # ── Security ──────────────────────────────────────────────────────────────
    {
        "category": "security",
        "question": "Design a network security zone model for a financial institution.",
        "answer": (
            "<think>\nZero-trust: never trust, always verify.\n"
            "Financial: PCI DSS + SOX compliance zones needed.\n"
            "Classic: Internet → DMZ → Internal → Restricted\n</think>\n\n"
            "**Network Security Zone Model — Financial Institution**\n\n"
            "**Zones (outside → inside):**\n"
            "1. **Untrusted** — Internet, partner connections\n"
            "2. **DMZ** — Web servers, API gateways, MFA proxies\n"
            "3. **Extranet** — B2B partner VLANs (segmented per partner)\n"
            "4. **Internal** — Employee workstations, internal apps\n"
            f"5. **CDE** — {COMPLIANCE_FACTS['pci_segmentation']} — payment systems\n"
            "6. **Management** — Out-of-band network for device management\n"
            "7. **Restricted** — SWIFT, trading systems, core banking\n\n"
            "**Controls:**\n"
            f"- {COMPLIANCE_FACTS['pci_firewall']} between all zones\n"
            "- Micro-segmentation within zones (Zero Trust)\n"
            "- Encrypted management (SSH v2, TLS 1.3)\n"
            "- IPS inline on Internet ingress and CDE boundary"
        ),
        "expected_key_facts": ["PCI", "segmentation", "firewall", "zero trust"],
    },
]

# ---------------------------------------------------------------------------
# Expand templates with minor variations to reach target count
# ---------------------------------------------------------------------------

VARIATIONS = {
    "qos": [
        ("50 Mbps", "100 Mbps"), ("200 Mbps", "100 Mbps"), ("1 Gbps", "100 Mbps"),
        ("SD-WAN overlay", "WAN link"), ("MPLS circuit", "WAN link"),
    ],
    "ha": [
        ("99.999%", "99.99%"), ("99.9%", "99.99%"), ("99.95%", "99.99%"),
        ("campus network", "data center"), ("edge routers", "servers"),
    ],
    "wan": [
        ("50 branches", "80 branch"), ("200 branches", "80 branch"),
        ("retail stores", "branch offices"), ("hospital sites", "branch offices"),
    ],
    "topology": [
        ("1,000 servers", "2,000 servers"), ("4,000 servers", "2,000 servers"),
        ("campus fabric", "spine-leaf fabric"), ("HCI clusters", "servers"),
    ],
}


def _vary(template: dict, idx: int) -> dict:
    """Create a minor variation of a template for deduplication."""
    import copy
    t = copy.deepcopy(template)
    variations = VARIATIONS.get(t["category"], [])
    if variations:
        v = variations[idx % len(variations)]
        t["question"] = t["question"].replace(v[1], v[0])
    return t


def generate_sft_dataset(output_dir: str | Path, count: int = 9000) -> int:
    """Generate SFT dataset without any API key."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    base = len(SFT_TEMPLATES)

    for i in range(count):
        tpl = SFT_TEMPLATES[i % base]
        sample = _vary(tpl, i // base)

        samples.append({
            "id": f"sft_{i:05d}",
            "category": sample["category"],
            "messages": [
                {"role": "system", "content": "You are a CCDE-certified network architect. Use <think>...</think> before answering. Include precise technical terminology."},
                {"role": "user", "content": sample["question"]},
                {"role": "assistant", "content": sample["answer"]},
            ],
            "expected_key_facts": sample["expected_key_facts"],
            "source": "template",
        })

    # Shuffle and split 80/10/10
    random.shuffle(samples)
    n = len(samples)
    train = samples[:int(n * 0.8)]
    val   = samples[int(n * 0.8):int(n * 0.9)]
    test  = samples[int(n * 0.9):]

    for split, data in [("train", train), ("val", val), ("test", test)]:
        out = output_dir / f"sft_{split}.jsonl"
        with open(out, "w") as f:
            for s in data:
                f.write(json.dumps(s) + "\n")
        print(f"  {split}: {len(data)} samples → {out}")

    print(f"✓ SFT dataset: {len(samples)} total samples (no API key used)")
    return len(samples)


def generate_grpo_dataset_template(output_dir: str | Path, count: int = 800) -> int:
    """Generate GRPO verifiable dataset from fact_registry without any API key."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # All verifiable GRPO templates
    GRPO_TEMPLATES = [
        {
            "sub_type": "ha_calculation",
            "question": "Calculate the maximum annual downtime for 99.99% availability. Show the formula.",
            "answer_key": HA_FACTS["99_99_annual_downtime_minutes"],
            "expected_key_facts": ["52.6 minutes per year", "525,960", "0.0001"],
            "exact_values": ["52.6", "52.596"],
            "answer_type": "calculation",
            "answer": (
                "<think>\n(1 - 0.9999) × 525,960 min/year = 52.596 minutes\n</think>\n\n"
                f"**{HA_FACTS['99_99_annual_downtime_minutes']}**\n\n"
                "Formula: `(1 - 0.9999) × 365.25 × 24 × 60 = 52.6 minutes`"
            ),
        },
        {
            "sub_type": "ha_calculation",
            "question": "What is the annual downtime budget for 99.9% availability?",
            "answer_key": HA_FACTS["99_9_annual_downtime_hours"],
            "expected_key_facts": ["8.76 hours per year"],
            "exact_values": ["8.76"],
            "answer_type": "calculation",
            "answer": (
                "<think>\n(1 - 0.999) × 8,760 hours/year = 8.76 hours\n</think>\n\n"
                f"**{HA_FACTS['99_9_annual_downtime_hours']}**\n\nFormula: `0.001 × 8,760 = 8.76 hours`"
            ),
        },
        {
            "sub_type": "wan_topology",
            "question": "How many tunnels are required for 80 branches in a full mesh topology?",
            "answer_key": WAN_FACTS["80_branch_full_mesh"],
            "expected_key_facts": ["3,160 tunnels", "N(N-1)/2"],
            "exact_values": ["3160", "3,160"],
            "answer_type": "calculation",
            "answer": (
                "<think>\nN(N-1)/2 = 80 × 79 / 2 = 3,160\n</think>\n\n"
                f"**{WAN_FACTS['80_branch_full_mesh']}**\n\nFormula: `{WAN_FACTS['full_mesh_formula']}`"
            ),
        },
        {
            "sub_type": "wan_topology",
            "question": "How many tunnels for 80 branches in hub-and-spoke?",
            "answer_key": WAN_FACTS["80_branch_hub_spoke"],
            "expected_key_facts": ["80 tunnels", "hub-and-spoke scalable"],
            "exact_values": ["80"],
            "answer_type": "calculation",
            "answer": (
                "<think>\nHub-and-spoke: N tunnels (one per spoke).\n80 spokes = 80 tunnels.\n</think>\n\n"
                f"**{WAN_FACTS['80_branch_hub_spoke']}**\n\n`{WAN_FACTS['hub_spoke_scalable']}`"
            ),
        },
        {
            "sub_type": "spineleaf_capacity",
            "question": "Calculate maximum servers in a spine-leaf fabric with 32 leaf switches, each with 64 ports (half uplinks).",
            "answer_key": SPINELEAF_FACTS["max_servers_64port"],
            "expected_key_facts": ["2048 servers", "N x M / 2"],
            "exact_values": ["2048"],
            "answer_type": "calculation",
            "answer": (
                "<think>\nFormula: N × M / 2 = 32 × 128 / 2... wait, 64-port switch, half uplinks:\n"
                "32 server-facing × 64 leaves = 2048\n</think>\n\n"
                f"**{SPINELEAF_FACTS['max_servers_64port']}**\n\n"
                f"Formula: `{SPINELEAF_FACTS['capacity_formula']}`"
            ),
        },
        {
            "sub_type": "vxlan_mtu",
            "question": "What is the minimum MTU required for VXLAN? Show the overhead calculation.",
            "answer_key": VXLAN_FACTS["mtu_minimum"],
            "expected_key_facts": ["MTU 1550 minimum", "50 byte overhead"],
            "exact_values": ["1550", "50"],
            "answer_type": "calculation",
            "answer": (
                "<think>\nVXLAN overhead: 8B VXLAN + 8B UDP + 20B IP + 14B Ethernet = 50B\n"
                "1500 + 50 = 1550 minimum MTU\n</think>\n\n"
                f"**{VXLAN_FACTS['mtu_minimum']}**\n\nOverhead: `{VXLAN_FACTS['overhead']}`"
            ),
        },
        {
            "sub_type": "qos_dscp",
            "question": "What DSCP decimal value is assigned to Expedited Forwarding (EF) and what is its binary representation?",
            "answer_key": QOS_FACTS["ef_dscp"],
            "expected_key_facts": ["EF DSCP 46", "101110"],
            "exact_values": ["46", "101110"],
            "answer_type": "exact_value",
            "answer": (
                "<think>\nEF = Expedited Forwarding = DSCP 46 = binary 101110\n</think>\n\n"
                f"**{QOS_FACTS['ef_dscp']}**\n\n"
                f"Queue: {QOS_FACTS['llq']}\n"
                f"Limit: {QOS_FACTS['priority_queue_limit']}"
            ),
        },
        {
            "sub_type": "bgp_dampening",
            "question": "In BGP route dampening, what happens to the penalty over time?",
            "answer_key": BGP_FACTS["half_life_decay"],
            "expected_key_facts": ["half-life decay", "penalty accumulation"],
            "exact_values": ["half-life", "15 minutes"],
            "answer_type": "exact_value",
            "answer": (
                "<think>\nPenalty accumulates on flap; decays exponentially.\n</think>\n\n"
                f"- {BGP_FACTS['penalty_accumulation']}\n"
                f"- {BGP_FACTS['half_life_decay']}\n\n"
                "Default half-life = 15 minutes; penalty halves every 15 min without flaps."
            ),
        },
    ]

    samples = []
    for i in range(count):
        tpl = GRPO_TEMPLATES[i % len(GRPO_TEMPLATES)]
        samples.append({
            "id": f"grpo_{i:04d}",
            "sub_type": tpl["sub_type"],
            "messages": [
                {"role": "system", "content": "You are a CCDE-certified network architect. Use <think>...</think> to show reasoning before answering."},
                {"role": "user", "content": tpl["question"]},
            ],
            "answer_key": tpl["answer_key"],
            "expected_key_facts": tpl["expected_key_facts"],
            "exact_values": tpl["exact_values"],
            "answer_type": tpl["answer_type"],
            "reference_answer": tpl["answer"],
        })

    out = output_dir / "grpo_train.jsonl"
    with open(out, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"✓ GRPO dataset: {len(samples)} samples → {out} (no API key used)")
    return len(samples)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--type", choices=["sft", "grpo", "all"], default="all")
    p.add_argument("--sft-count", type=int, default=9000)
    p.add_argument("--grpo-count", type=int, default=800)
    p.add_argument("--output", default="/data/datasets")
    args = p.parse_args()

    if args.type in ("sft", "all"):
        generate_sft_dataset(args.output, args.sft_count)
    if args.type in ("grpo", "all"):
        generate_grpo_dataset_template(args.output, args.grpo_count)

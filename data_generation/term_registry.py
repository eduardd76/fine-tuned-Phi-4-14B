"""
term_registry.py — Required CCDE terminology per question sub-type.
Maps question sub_type strings to lists of exact phrases the model MUST use.
Used by quality_validator.py to reject LLM-generated training samples that
use paraphrased terminology instead of the exact CCDE terms.
"""

REQUIRED_TERMS: dict = {

    # ── HA & SLA ──────────────────────────────────────────────────────────────
    "ha_sla_calculation": [
        "52.6 minutes per year", "99.99%", "four nines",
    ],
    "ha_redundancy_model": [
        "N+1 optimal", "N+2 diminishing returns", "50% utilization limit",
        "active-active complexity",
    ],
    "fhrp_comparison": [
        "GLBP active-active", "AVG AVF",
        "HSRP disabled preemption", "VRRP open standard",
    ],

    # ── Data Center Topology ──────────────────────────────────────────────────
    "spineleaf_capacity": [
        "N x M / 2", "2048 servers", "8192 servers",
    ],
    "spineleaf_vs_threetier": [
        "VM mobility", "consistent latency", "horizontal scaling",
        "north-south for three-tier",
    ],
    "vxlan_design": [
        "symmetric IRB", "L3VNI", "distributed anycast gateway",
        "BGP EVPN control plane",
    ],
    "vxlan_mtu": [
        "50 byte overhead", "MTU 1550 minimum", "VTEP requirement",
    ],

    # ── WAN ───────────────────────────────────────────────────────────────────
    "wan_topology": [
        "N(N-1)/2 formula", "hub-and-spoke scalable",
        "full mesh expensive", "partial mesh compromise",
    ],

    # ── QoS ───────────────────────────────────────────────────────────────────
    "qos_voip_wan": [
        "EF DSCP 46", "LLQ", "33% priority queue limit", "trust boundary",
    ],
    "qos_shaping_policing": [
        "shaping buffers delays", "TCP retransmissions", "egress only for shaping",
    ],

    # ── BGP ───────────────────────────────────────────────────────────────────
    "bgp_multihoming_same_asn": [
        "AS-override", "allowas-in", "loop prevention", "same ASN multi-homing",
    ],
    "bgp_dampening": [
        "penalty accumulation", "flap count", "suppress limit",
        "half-life decay", "reuse limit",
    ],
    "bgp_ibgp_nexthop": [
        "IGP reachability", "ibgp next-hop unchanged",
    ],

    # ── OSPF ──────────────────────────────────────────────────────────────────
    "ospf_multiarea_design": [
        "multi-area", "area 0 backbone", "ABR summarization",
        "SPF calculation", "type 3 LSA",
    ],
    "ospf_stub_areas": [
        "blocks type 3 4 5 LSAs", "default route from ABR", "smallest LSDB",
    ],

    # ── MPLS ──────────────────────────────────────────────────────────────────
    "mpls_l3vpn_rd_rt": [
        "RD uniqueness", "RT import export", "VRF isolation", "overlapping addresses",
    ],
    "mpls_inter_as": [
        "Option A most secure back-to-back VRF", "Option C most scalable",
        "Option B balanced", "ASBR requirements",
    ],

    # ── SD-WAN ────────────────────────────────────────────────────────────────
    "sdwan_architecture": [
        "vBond orchestration", "vManage management",
        "vSmart control OMP", "vEdge data plane", "BFD SLA",
    ],
    "sdwan_path_selection": [
        "BFD probes", "SLA class", "brownout detection",
    ],

    # ── Compliance & Security ─────────────────────────────────────────────────
    "pci_cde_design": [
        "CDE isolation", "dedicated firewall", "no direct internet to CDE",
    ],
    "dmz_architecture": [
        "DMZ buffer zone", "two-tier DMZ", "web server front-end",
        "database backend separation", "stateful inspection",
    ],
    "hipaa_segmentation": [
        "VRF isolation", "PVLAN", "medical device VLAN",
    ],

    # ── Design Methodology ────────────────────────────────────────────────────
    "ccde_methodology": [
        "top-down design", "business requirements", "technical constraints",
        "organizational constraints", "application criticality",
    ],

    # ── Routing Protocol Selection ────────────────────────────────────────────
    "igp_vs_egp_selection": [
        "OSPF for single-vendor campus", "BGP for multi-AS",
        "interior vs exterior", "topology hiding",
    ],
}


def get_required_terms(sub_type: str) -> list:
    """Return required exact terms for a question sub_type. Returns [] if unknown."""
    return REQUIRED_TERMS.get(sub_type, [])


def get_all_sub_types() -> list:
    return list(REQUIRED_TERMS.keys())

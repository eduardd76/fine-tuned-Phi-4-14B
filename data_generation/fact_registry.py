"""
fact_registry.py — Hardcoded ground-truth facts for GRPO verifiable rewards.
These values are mathematically provable or from Cisco documentation.
Never generate these with an LLM — LLM hallucination would corrupt ground truth.
"""

# ── SLA / HA Calculations ─────────────────────────────────────────────────────
HA_FACTS = {
    "99_99_annual_downtime_minutes": "52.6 minutes per year",
    "99_99_annual_downtime_seconds": "3,156 seconds per year",
    "99_9_annual_downtime_hours": "8.76 hours per year",
    "n_plus_1_rule": "N+1 optimal — adds one redundant unit, cost-effective",
    "n_plus_2_rule": "N+2 diminishing returns — doubles redundancy cost for marginal gain",
    "active_active_utilization": "50% utilization limit — each node must handle full load on failover",
    "n_plus_1_minimum": "N+1 minimum — industry standard for carrier-grade availability",
    "convergence_complexity": "convergence complexity — more redundancy increases reconvergence time",
}

# ── Spine-Leaf / Data Center Topology ────────────────────────────────────────
SPINELEAF_FACTS = {
    "max_servers_64port": "2048 servers — 32 leaf × 64 downlinks / 2 (half uplinks)",
    "max_servers_128port": "8192 servers — 64 leaf × 128 downlinks / 2",
    "capacity_formula": "N x M / 2 — where N=leaf count, M=server-facing ports per leaf",
    "oversubscription_formula": "oversubscription = downlink_ports / uplink_ports",
    "three_tier_webscale": "three-tier clos for web-scale — when spine-leaf max density exceeded",
    "consistent_latency": "consistent latency — spine-leaf provides equal hop count for any A-to-B path",
    "vm_mobility": "VM mobility — spine-leaf VXLAN enables live migration across pods",
    "horizontal_scaling": "horizontal scaling — add leaf switches without redesigning spine",
    "north_south_three_tier": "north-south for three-tier — traditional 3-tier optimized for client-server",
}

# ── WAN / Mesh Topology ───────────────────────────────────────────────────────
WAN_FACTS = {
    "full_mesh_formula": "N(N-1)/2 formula — links required for full mesh",
    "hub_spoke_scalable": "hub-and-spoke scalable — O(N) links, single failure point at hub",
    "full_mesh_expensive": "full mesh expensive — N(N-1)/2 grows quadratically",
    "partial_mesh_compromise": "partial mesh compromise — balances redundancy and cost",
    "80_branch_full_mesh": "3,160 tunnels for 80-branch full mesh (80×79/2)",
    "80_branch_hub_spoke": "80 tunnels for 80-branch hub-and-spoke",
}

# ── VXLAN / EVPN ─────────────────────────────────────────────────────────────
VXLAN_FACTS = {
    "overhead": "50 byte overhead — 8B VXLAN + 8B UDP + 20B outer IP + 14B outer Ethernet",
    "mtu_minimum": "MTU 1550 minimum — 1500 payload + 50B VXLAN overhead",
    "vtep_requirement": "VTEP requirement — every leaf must run VXLAN Tunnel Endpoint",
    "symmetric_irb": "symmetric IRB — both ingress and egress VTEP do L3 lookup, requires L3VNI",
    "l3vni": "L3VNI — per-VRF VNI used for inter-subnet routing in symmetric IRB",
    "distributed_anycast_gateway": "distributed anycast gateway — same IP/MAC on all leaf switches for default gateway",
    "bgp_evpn_control_plane": "BGP EVPN control plane — distributes MAC/IP bindings without flooding",
    "vni_range": "VNI range 1-16777215 (24-bit field)",
}

# ── QoS ───────────────────────────────────────────────────────────────────────
QOS_FACTS = {
    "ef_dscp": "EF DSCP 46 — Expedited Forwarding, decimal 46, binary 101110",
    "llq": "LLQ — Low Latency Queuing, strict priority queue for voice/video",
    "priority_queue_limit": "33% priority queue limit — prevents starvation of other traffic classes",
    "trust_boundary": "trust boundary — point where DSCP markings are trusted/re-marked",
    "cs1_dscp": "CS1 DSCP 8 — scavenger class, lower than best-effort",
    "af41_dscp": "AF41 DSCP 34 — video conferencing class",
    "dscp_classes": "EF=46 voice, AF4x=video, AF3x=critical data, CS0=best-effort",
    "shaping_buffers": "shaping buffers delays — traffic is buffered and released at shaped rate",
    "tcp_retransmissions": "TCP retransmissions — policing drops cause TCP to retransmit, shaping avoids this",
    "egress_shaping": "egress only for shaping — shaping requires egress queue, cannot shape ingress",
}

# ── BGP ───────────────────────────────────────────────────────────────────────
BGP_FACTS = {
    "as_override": "AS-override — replace customer ASN in AS_PATH with provider ASN to prevent loop",
    "allowas_in": "allowas-in — permit receiving routes with own ASN in AS_PATH (same-ASN multi-homing)",
    "loop_prevention": "loop prevention — AS_PATH attribute; router rejects routes containing own ASN",
    "same_asn_multihoming": "same ASN multi-homing — requires AS-override on provider or allowas-in on CE",
    "igp_reachability": "IGP reachability — IBGP next-hop must be reachable via IGP",
    "ibgp_next_hop": "ibgp next-hop unchanged — IBGP does not change next-hop; IGP must resolve it",
    "route_reflector": "route reflector — eliminates IBGP full-mesh requirement",
    "dampening_penalty": "penalty accumulation — each flap adds 1000 to penalty",
    "suppress_limit": "suppress limit 2000 — route suppressed when penalty exceeds 2000",
    "half_life": "half-life decay — penalty halves every 15 minutes (default)",
    "reuse_limit": "reuse limit 750 — route unsuppressed when penalty drops below 750",
    "flap_count": "flap count — number of times route changed state within decay window",
}

# ── OSPF ──────────────────────────────────────────────────────────────────────
OSPF_FACTS = {
    "multi_area": "multi-area — divide large OSPF domain to reduce SPF scope and LSA flooding",
    "area_0_backbone": "area 0 backbone — all areas must connect to area 0; inter-area traffic transits backbone",
    "abr_summarization": "ABR summarization — Area Border Router injects Type 3 summary LSAs",
    "spf_calculation": "SPF calculation — Dijkstra runs per-area; multi-area limits SPF scope",
    "type_3_lsa": "type 3 LSA — summary LSA from ABR describing inter-area routes",
    "totally_stubby": "totally stubby area — blocks type 3 4 5 LSAs; only default route from ABR",
    "blocks_lsas": "blocks type 3 4 5 LSAs — NSSA blocks 5; totally stubby blocks 3, 4, 5",
    "default_route_abr": "default route from ABR — ABR injects 0.0.0.0/0 into stub/totally-stubby area",
    "smallest_lsdb": "smallest LSDB — totally stubby area has smallest LSDB; only intra-area + default",
    "dead_timer": "OSPF dead timer 40 seconds — Cisco IOS default (4× hello interval)",
}

# ── FHRP (First Hop Redundancy) ───────────────────────────────────────────────
FHRP_FACTS = {
    "glbp_active_active": "GLBP active-active — AVG assigns different AVFs to load-balance across gateways",
    "hsrp_preemption": "HSRP disabled preemption — preemption off by default; active stays active after recovery",
    "vrrp_open_standard": "VRRP open standard — RFC 5798; HSRP is Cisco proprietary",
    "avg_avf": "AVG AVF — Active Virtual Gateway assigns Active Virtual Forwarders in GLBP",
    "hsrp_priority": "HSRP priority 100 default — higher wins; preempt must be enabled for switchback",
}

# ── SD-WAN ────────────────────────────────────────────────────────────────────
SDWAN_FACTS = {
    "vbond": "vBond orchestration — authenticates vEdge, helps NAT traversal",
    "vmanage": "vManage management — centralized GUI/API for policy and monitoring",
    "vsmart_omp": "vSmart control OMP — OMP (Overlay Management Protocol) distributes routes/policies",
    "vedge_dataplane": "vEdge data plane — performs packet forwarding, encryption, BFD",
    "bfd_sla": "BFD SLA — BFD probes measure latency/jitter/loss per tunnel for SLA policy",
    "bfd_probes": "BFD probes — sub-second link quality detection for path selection",
    "sla_class": "SLA class — policy matching traffic to tunnels meeting latency/loss thresholds",
    "brownout_detection": "brownout detection — BFD detects degraded link before full failure",
}

# ── MPLS ──────────────────────────────────────────────────────────────────────
MPLS_FACTS = {
    "rd_uniqueness": "RD uniqueness — Route Distinguisher makes VPN routes globally unique in BGP",
    "rt_import_export": "RT import export — Route Target controls which VRFs import/export routes",
    "vrf_isolation": "VRF isolation — separate routing tables prevent inter-customer leakage",
    "overlapping_addresses": "overlapping addresses — RD makes identical prefixes unique across VPNs",
    "option_a": "Option A most secure back-to-back VRF — re-originates all routes at ASBR boundary",
    "option_b": "Option B balanced — exchanges labeled VPN routes between ASBRs",
    "option_c": "Option C most scalable — recursive BGP next-hop, no VRF state at ASBR",
    "asbr_requirements": "ASBR requirements — Option A: VRF per customer; Option B: eBGP labeled; Option C: recursive",
}

# ── Compliance / Security ─────────────────────────────────────────────────────
COMPLIANCE_FACTS = {
    "cde_isolation": "CDE isolation — Cardholder Data Environment must be isolated per PCI DSS",
    "dedicated_firewall": "dedicated firewall — CDE requires dedicated firewall, not shared ACL",
    "no_direct_internet": "no direct internet to CDE — all CDE traffic must traverse firewall/IPS",
    "dmz_buffer_zone": "DMZ buffer zone — screened subnet between internet and internal network",
    "two_tier_dmz": "two-tier DMZ — outer firewall (internet→DMZ) + inner firewall (DMZ→internal)",
    "web_server_frontend": "web server front-end — web tier in DMZ, no direct DB access",
    "database_backend": "database backend separation — database behind inner firewall, never in DMZ",
    "stateful_inspection": "stateful inspection — tracks connection state; default-deny on new inbound",
    "vrf_isolation_hipaa": "VRF isolation — HIPAA medical device segmentation via separate VRF",
    "pvlan": "PVLAN — Private VLAN isolates devices within same subnet (e.g., medical devices)",
    "medical_device_vlan": "medical device VLAN — dedicated VLAN with ACL for FDA-regulated devices",
}

# ── Design Methodology ────────────────────────────────────────────────────────
METHODOLOGY_FACTS = {
    "top_down_design": "top-down design — start with business requirements, then technical, then implementation",
    "business_requirements": "business requirements — budget, SLA, growth projections, compliance",
    "technical_constraints": "technical constraints — existing infrastructure, protocol support, vendor",
    "organizational_constraints": "organizational constraints — staff skills, support model, change windows",
    "application_criticality": "application criticality — classify apps by tier for QoS and HA design",
}

# ── Master registry by topic ──────────────────────────────────────────────────
ALL_FACTS = {
    "ha": HA_FACTS,
    "spineleaf": SPINELEAF_FACTS,
    "wan": WAN_FACTS,
    "vxlan": VXLAN_FACTS,
    "qos": QOS_FACTS,
    "bgp": BGP_FACTS,
    "ospf": OSPF_FACTS,
    "fhrp": FHRP_FACTS,
    "sdwan": SDWAN_FACTS,
    "mpls": MPLS_FACTS,
    "compliance": COMPLIANCE_FACTS,
    "methodology": METHODOLOGY_FACTS,
}


def get_facts_for_topic(topic: str) -> dict:
    return ALL_FACTS.get(topic, {})


def get_required_terms_for_question(topics: list) -> list:
    """Return all exact phrases required for a given set of topics."""
    terms = []
    for topic in topics:
        terms.extend(ALL_FACTS.get(topic, {}).values())
    return terms

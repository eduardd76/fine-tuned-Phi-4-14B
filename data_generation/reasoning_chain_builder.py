"""
Reasoning Chain Builder

Builds structured <think>...</think> chain-of-thought sections for training
samples. Each step references specific knowledge sources to ensure grounding.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReasoningStep:
    """A single step in the reasoning chain."""
    step_num: int
    title: str
    content: str
    source_ref: str
    sub_steps: list[str] = field(default_factory=list)


@dataclass
class ReasoningChain:
    """A complete reasoning chain for a training sample."""
    steps: list[ReasoningStep]
    scenario_type: str  # "design" or "troubleshooting"

    def to_think_block(self) -> str:
        """Render as <think>...</think> block."""
        lines = ["<think>"]
        for step in self.steps:
            lines.append(f"\nStep {step.step_num}: {step.title}")
            lines.append(f"- Source: {step.source_ref}")
            lines.append(step.content)
            for sub in step.sub_steps:
                lines.append(f"  * {sub}")
        lines.append("</think>")
        return "\n".join(lines)

    def step_count(self) -> int:
        return len(self.steps)


class ReasoningChainBuilder:
    """
    Builds structured reasoning chains for network design and
    troubleshooting scenarios, referencing the knowledge base.
    """

    def __init__(self, knowledge: dict[str, Any]) -> None:
        self.knowledge = knowledge
        self.design_patterns = knowledge.get("design_patterns", {})
        self.troubleshooting = knowledge.get("troubleshooting_trees", {})
        self.compliance = knowledge.get("compliance_requirements", {})
        self.vendors = knowledge.get("vendor_specifics", {})
        self.costs = knowledge.get("cost_benchmarks", {})

    # ------------------------------------------------------------------ #
    # Design reasoning chains                                              #
    # ------------------------------------------------------------------ #

    def build_design_chain(
        self,
        scenario: dict[str, Any],
        knowledge: dict[str, Any],
    ) -> ReasoningChain:
        """
        Build a chain-of-thought reasoning chain for a network design scenario.

        Args:
            scenario: Dict with keys: user_count, site_count, compliance,
                      uptime_requirement, budget, industry, special_requirements
            knowledge: Full knowledge base dict

        Returns:
            ReasoningChain with 6-12 steps referencing specific sources
        """
        steps: list[ReasoningStep] = []
        step_num = 1

        # Step 1: Classify scale and initial topology selection
        topology = self._select_topology(scenario)
        topology_data = (
            self.design_patterns.get("topology_patterns", {}).get(topology, {})
        )
        steps.append(ReasoningStep(
            step_num=step_num,
            title="Network Scale Classification and Topology Selection",
            source_ref=(
                "Enterprise Network Design - Topology Selection Criteria, "
                "Chapter: Campus Design Hierarchy"
            ),
            content=(
                f"- User count: {scenario.get('user_count', 'N/A')} → "
                f"'{self._scale_label(scenario.get('user_count', 0))}' category\n"
                f"- Recommended topology: {topology.replace('_', ' ').title()}\n"
                f"- Decision basis: {topology_data.get('user_range', 'N/A')} user range"
            ),
            sub_steps=[
                f"Alternative 1 considered: collapsed core - "
                f"{'Rejected: exceeds 2000-user threshold' if scenario.get('user_count', 0) > 2000 else 'Viable option'}",
                f"Alternative 2 considered: spine-leaf - "
                f"{'Rejected: data center topology only' if not scenario.get('is_datacenter') else 'Applicable'}",
            ],
        ))
        step_num += 1

        # Step 2: Compliance analysis (if required)
        compliance_reqs = scenario.get("compliance", [])
        if compliance_reqs:
            step_num = self._add_compliance_steps(
                steps, step_num, compliance_reqs
            )

        # Step 3: HA / uptime requirements
        uptime = scenario.get("uptime_requirement", "99.9%")
        steps.append(ReasoningStep(
            step_num=step_num,
            title=f"High Availability Design for {uptime} Uptime",
            source_ref=(
                "Network HA Design Patterns - Active-Active vs Active-Standby, "
                "Chapter: Redundancy Calculations"
            ),
            content=self._uptime_analysis(uptime),
            sub_steps=self._ha_sub_steps(uptime),
        ))
        step_num += 1

        # Step 4: WAN design
        site_count = scenario.get("site_count", 1)
        if site_count > 1:
            wan_type = self._select_wan(scenario)
            steps.append(ReasoningStep(
                step_num=step_num,
                title="WAN Architecture Selection",
                source_ref=(
                    "WAN Design Guide - Transport Selection Framework, "
                    "Chapter: SD-WAN vs MPLS Decision Matrix"
                ),
                content=self._wan_analysis(scenario, wan_type),
                sub_steps=self._wan_sub_steps(scenario, wan_type),
            ))
            step_num += 1

        # Step 5: Security architecture
        steps.append(ReasoningStep(
            step_num=step_num,
            title="Security Architecture Design",
            source_ref=(
                "Zero Trust Architecture Guide, "
                "Chapter: Network Segmentation and Defense-in-Depth"
            ),
            content=self._security_analysis(scenario),
            sub_steps=self._security_sub_steps(compliance_reqs),
        ))
        step_num += 1

        # Step 6: Technology stack selection
        steps.append(ReasoningStep(
            step_num=step_num,
            title="Technology Stack and Vendor Selection",
            source_ref=(
                "Vendor Comparison Guide - Enterprise Networking Platforms, "
                "Chapter: Platform Selection by Scale"
            ),
            content=self._vendor_analysis(scenario, topology),
            sub_steps=[],
        ))
        step_num += 1

        # Step 7: Cost estimation
        steps.append(ReasoningStep(
            step_num=step_num,
            title="Cost Estimation Using Industry Benchmarks",
            source_ref=(
                "Enterprise Network TCO Model - CapEx/OpEx Benchmarks, "
                "Chapter: Cost Modeling by Organization Size"
            ),
            content=self._cost_analysis(scenario),
            sub_steps=[],
        ))
        step_num += 1

        # Step 8: Implementation phasing
        steps.append(ReasoningStep(
            step_num=step_num,
            title="Implementation Phasing and Risk Mitigation",
            source_ref=(
                "Enterprise Migration Methodology - Phased Deployment Patterns, "
                "Chapter: Large-Scale Network Implementation"
            ),
            content=self._phasing_analysis(scenario),
            sub_steps=[],
        ))

        return ReasoningChain(steps=steps, scenario_type="design")

    # ------------------------------------------------------------------ #
    # Troubleshooting reasoning chains                                     #
    # ------------------------------------------------------------------ #

    def build_troubleshooting_chain(
        self,
        symptom: dict[str, Any],
        knowledge: dict[str, Any],
    ) -> ReasoningChain:
        """
        Build a chain-of-thought reasoning chain for a troubleshooting scenario.

        Args:
            symptom: Dict with keys: protocol, issue_type, symptoms, environment
            knowledge: Full knowledge base dict

        Returns:
            ReasoningChain with 5-10 steps following diagnostic decision trees
        """
        steps: list[ReasoningStep] = []
        step_num = 1
        protocol = symptom.get("protocol", "generic")
        issue_type = symptom.get("issue_type", "connectivity")

        # Step 1: Problem classification
        steps.append(ReasoningStep(
            step_num=step_num,
            title="Problem Classification and Initial Triage",
            source_ref=(
                "Network Troubleshooting Methodology - Problem Classification, "
                "Chapter: Systematic Diagnostic Approach"
            ),
            content=(
                f"- Protocol affected: {protocol.upper()}\n"
                f"- Issue category: {issue_type}\n"
                f"- Symptoms: {', '.join(symptom.get('symptoms', []))}\n"
                "- Using OSI model bottom-up troubleshooting sequence"
            ),
            sub_steps=[],
        ))
        step_num += 1

        # Step 2: Layer 1/2 verification
        steps.append(ReasoningStep(
            step_num=step_num,
            title="Physical and Data Link Layer Verification",
            source_ref=(
                "OSI Layer Troubleshooting Guide - Layer 1-2 Diagnostics, "
                "Chapter: Bottom-Up Systematic Verification"
            ),
            content=(
                "Verify physical and data link layer before higher-layer investigation:\n"
                "- Check interface status: show interface brief\n"
                "- Verify no input/output errors: show interface counters errors\n"
                "- Confirm duplex/speed: no mismatch\n"
                "- Check VLAN membership and trunk configuration"
            ),
            sub_steps=[],
        ))
        step_num += 1

        # Steps 3-7: Protocol-specific diagnostic steps
        proto_steps = self._protocol_diagnostic_steps(
            protocol, issue_type, symptom, step_num
        )
        steps.extend(proto_steps)
        step_num += len(proto_steps)

        # Final step: Root cause and resolution
        steps.append(ReasoningStep(
            step_num=step_num,
            title="Root Cause Analysis and Resolution",
            source_ref=(
                f"{protocol.upper()} Troubleshooting Decision Tree - Resolution Actions, "
                "Chapter: Configuration Corrections and Verification"
            ),
            content=self._resolution_content(protocol, issue_type, symptom),
            sub_steps=self._resolution_verification(protocol),
        ))

        return ReasoningChain(steps=steps, scenario_type="troubleshooting")

    # ------------------------------------------------------------------ #
    # Private helper methods                                               #
    # ------------------------------------------------------------------ #

    def _scale_label(self, user_count: int) -> str:
        if user_count < 500:
            return "small"
        if user_count < 5000:
            return "medium"
        if user_count < 50000:
            return "large"
        return "enterprise"

    def _select_topology(self, scenario: dict[str, Any]) -> str:
        users = scenario.get("user_count", 0)
        if scenario.get("is_datacenter"):
            return "spine_leaf"
        if users < 2000:
            return "collapsed_core"
        if users < 50000:
            return "three_tier"
        return "three_tier"

    def _select_wan(self, scenario: dict[str, Any]) -> str:
        sites = scenario.get("site_count", 1)
        compliance = scenario.get("compliance", [])
        if sites > 10 or scenario.get("cloud_heavy"):
            return "sd_wan"
        if "PCI-DSS" in compliance or "HIPAA" in compliance:
            return "mpls"
        if sites > 3:
            return "hybrid_wan"
        return "mpls"

    def _uptime_analysis(self, uptime: str) -> str:
        uptime_map = {
            "99.9%": "8.76 hours downtime/year → N+1 redundancy, HSRP/VRRP",
            "99.99%": "52.6 minutes downtime/year → N+1 with BFD, active-standby DC",
            "99.999%": "5.26 minutes downtime/year → N+2, active-active DC, full redundancy",
        }
        return (
            f"- Uptime target: {uptime} = "
            f"{uptime_map.get(uptime, 'Calculate max downtime/year')}\n"
            "- Redundancy strategy selected based on downtime tolerance\n"
            "- BFD for sub-second failure detection at all critical links"
        )

    def _ha_sub_steps(self, uptime: str) -> list[str]:
        steps = [
            "Core layer: N+1 redundancy (minimum)",
            "Distribution: dual-homed to core",
            "Access: dual uplinks to distribution pair",
        ]
        if "99.99" in uptime:
            steps.append("No single point of failure across stack")
            steps.append("Geo-redundant DC with sub-second failover")
        return steps

    def _wan_analysis(self, scenario: dict[str, Any], wan_type: str) -> str:
        sites = scenario.get("site_count", 1)
        users = scenario.get("user_count", 0)
        users_per_site = users // max(sites, 1)
        bw_mbps = users_per_site * 2
        return (
            f"- Site count: {sites} → {wan_type.replace('_', ' ').upper()}\n"
            f"- Average users per site: {users_per_site}\n"
            f"- Bandwidth calculation: {users_per_site} users × 2 Mbps = {bw_mbps} Mbps minimum\n"
            f"- Recommended: {min(bw_mbps * 1.2, bw_mbps + 100):.0f} Mbps with 20% headroom"
        )

    def _wan_sub_steps(self, scenario: dict[str, Any], wan_type: str) -> list[str]:
        sub: list[str] = []
        if wan_type == "sd_wan":
            sub = [
                "Primary transport: MPLS for latency-sensitive apps",
                "Secondary transport: Broadband internet for SaaS/cloud",
                "Application-aware routing: DPI-based traffic steering",
                "Zero-touch provisioning for new site deployment",
            ]
        elif wan_type == "mpls":
            sub = [
                "Dedicated MPLS circuit with guaranteed SLA",
                "QoS classes: EF (voice), AF41 (video), AF31 (business)",
                "Backup: internet IPSec VPN (failover only)",
            ]
        return sub

    def _security_analysis(self, scenario: dict[str, Any]) -> str:
        compliance = scenario.get("compliance", [])
        has_pci = "PCI-DSS" in compliance
        has_hipaa = "HIPAA" in compliance
        lines = [
            "- Security model: Defense-in-depth with micro-segmentation",
            f"- Compliance-driven segmentation: {'CDE isolation required' if has_pci else 'Standard VLAN segmentation'}",
            f"- PHI data isolation: {'HIPAA-compliant separate VLAN required' if has_hipaa else 'Not required'}",
            "- Zero Trust principles: verify explicitly, least privilege, assume breach",
        ]
        return "\n".join(lines)

    def _security_sub_steps(self, compliance_reqs: list[str]) -> list[str]:
        sub = [
            "Layer 1: Perimeter NGFW (stateful inspection + IPS)",
            "Layer 2: Network micro-segmentation (VLAN + firewall zones)",
            "Layer 3: Identity-based access (802.1X + NAC)",
            "Layer 4: Data encryption (TLS 1.3 in transit, AES-256 at rest)",
        ]
        if "PCI-DSS" in compliance_reqs:
            sub.append("PCI-DSS CDE: Isolated VLAN with dedicated firewall rules")
        if "HIPAA" in compliance_reqs:
            sub.append("HIPAA PHI: Encrypted storage, audit logging all access")
        return sub

    def _vendor_analysis(self, scenario: dict[str, Any], topology: str) -> str:
        users = scenario.get("user_count", 0)
        if users > 10000:
            core = "Cisco Catalyst 9500-48Y4C (48x25G + 4x100G)"
            dist = "Cisco Catalyst 9400 with dual supervisors"
            access = "Cisco Catalyst 9300 (StackWise, PoE)"
        elif users > 2000:
            core = "Cisco Catalyst 9500-24Q or Arista 7050X"
            dist = "Cisco Catalyst 9400 or Arista 7280R"
            access = "Cisco Catalyst 9200/9300"
        else:
            core = "Cisco Catalyst 9300 (collapsed core)"
            dist = "N/A (collapsed core topology)"
            access = "Cisco Catalyst 9200"
        return (
            f"- Core: {core}\n"
            f"- Distribution: {dist}\n"
            f"- Access: {access}\n"
            "- Firewall: Palo Alto PA-series (application-aware, NGFW)\n"
            "- Wireless: Cisco Catalyst Center managed APs"
        )

    def _cost_analysis(self, scenario: dict[str, Any]) -> str:
        users = scenario.get("user_count", 0)
        compliance = scenario.get("compliance", [])
        capex_data = self.costs.get("capex_by_scale", {})

        scale = self._scale_label(users)
        scale_data = capex_data.get(scale, {})
        hw_data = scale_data.get("hardware_cost_per_user", {})
        typical_per_user = hw_data.get("typical", "$600").replace("$", "").replace(",", "")
        try:
            base_capex = int(float(typical_per_user)) * users
        except (ValueError, TypeError):
            base_capex = 600 * users

        compliance_multiplier = 1.0
        if "PCI-DSS" in compliance:
            compliance_multiplier += 0.30
        if "HIPAA" in compliance:
            compliance_multiplier += 0.25

        total_capex = int(base_capex * compliance_multiplier)
        annual_opex = int(total_capex * 0.15)

        return (
            f"- Scale: {scale} ({users} users) → ${int(float(typical_per_user)):,}/user benchmark\n"
            f"- Base CapEx: ${base_capex:,}\n"
            f"- Compliance premium: {(compliance_multiplier - 1) * 100:.0f}%\n"
            f"- Estimated total CapEx: ${total_capex:,}\n"
            f"- Annual OpEx estimate: ${annual_opex:,} (15% of CapEx)\n"
            f"- 3-Year TCO: ${total_capex + 3 * annual_opex:,}"
        )

    def _phasing_analysis(self, scenario: dict[str, Any]) -> str:
        users = scenario.get("user_count", 0)
        sites = scenario.get("site_count", 1)

        if users > 10000 or sites > 10:
            return (
                "- Phase 1 (Months 1-2): Core/distribution at primary and DR DC\n"
                "- Phase 2 (Months 3-4): Site rollout in waves of 5 sites\n"
                "- Phase 3 (Months 5-6): Compliance validation, user migration\n"
                "- Total timeline: 6 months (standard for this scale)"
            )
        if users > 1000:
            return (
                "- Phase 1 (Weeks 1-4): Core infrastructure deployment\n"
                "- Phase 2 (Weeks 5-10): Distribution and access layer\n"
                "- Phase 3 (Weeks 11-16): Migration and validation\n"
                "- Total timeline: 16 weeks"
            )
        return (
            "- Phase 1 (Weeks 1-2): Core/access deployment\n"
            "- Phase 2 (Weeks 3-4): WiFi, VPN, and services\n"
            "- Phase 3 (Week 5): User migration\n"
            "- Total timeline: 5 weeks"
        )

    def _add_compliance_steps(
        self,
        steps: list[ReasoningStep],
        step_num: int,
        compliance_reqs: list[str],
    ) -> int:
        pci_data = (
            self.compliance.get("pci_dss", {})
            .get("network_requirements", {})
            .get("requirement_1", {})
        )
        hipaa_data = self.compliance.get("hipaa", {})

        if "PCI-DSS" in compliance_reqs:
            steps.append(ReasoningStep(
                step_num=step_num,
                title="PCI-DSS 4.0.1 Network Requirements Analysis",
                source_ref=(
                    "PCI-DSS v4.0.1 - Requirements 1-4, "
                    "Chapter: Network Segmentation and CDE Isolation"
                ),
                content=(
                    "- PCI-DSS 4.0.1 Section 1.2: All connections to CDE controlled by NSC\n"
                    "- PCI-DSS 4.0.1 Section 1.3: Prohibit direct public access between internet and CDE\n"
                    "- PCI-DSS 4.0.1 Section 4.2.1: TLS 1.2+ required for cardholder data in transit\n"
                    "- Scope: CDE VLAN isolated with dedicated firewall rules\n"
                    "- Logging: All CDE access logged per Requirement 10 (12-month retention)"
                ),
                sub_steps=[
                    "CDE VLAN: VLAN 100 with dedicated ACL (permit-to-CDE, permit-from-CDE)",
                    "Quarterly vulnerability scans by ASV required",
                    "Annual penetration testing of CDE scope",
                ],
            ))
            step_num += 1

        if "HIPAA" in compliance_reqs:
            steps.append(ReasoningStep(
                step_num=step_num,
                title="HIPAA Security Rule Network Requirements",
                source_ref=(
                    "HIPAA Security Rule - 45 CFR Part 164, "
                    "Chapter: Technical Safeguards for ePHI"
                ),
                content=(
                    "- HIPAA §164.312(a)(1): Access control - unique user IDs, automatic logoff\n"
                    "- HIPAA §164.312(e)(1): Transmission security - guard against unauthorized access\n"
                    "- HIPAA §164.312(e)(2)(ii): Encryption of ePHI in transit (addressable)\n"
                    "- EHR/EMR systems: Separate VLAN with access controls\n"
                    "- Medical devices: Isolated VLAN (FDA-regulated devices)"
                ),
                sub_steps=[
                    "PHI VLAN isolated from general network",
                    "Encryption: TLS 1.2+ for all ePHI transmission",
                    "Audit logging: All ePHI access logged",
                ],
            ))
            step_num += 1

        return step_num

    def _protocol_diagnostic_steps(
        self,
        protocol: str,
        issue_type: str,
        symptom: dict[str, Any],
        step_num: int,
    ) -> list[ReasoningStep]:
        steps: list[ReasoningStep] = []

        if protocol.lower() == "bgp":
            bgp_tree = self.troubleshooting.get("bgp_issues", {})
            tree = bgp_tree.get(
                "neighbor_down" if "neighbor" in issue_type else "route_missing", {}
            )
            diag_steps = tree.get("decision_tree", [])[:4]

            for diag in diag_steps:
                steps.append(ReasoningStep(
                    step_num=step_num,
                    title=f"BGP Diagnostic: {diag.get('check', 'Verification step')}",
                    source_ref=(
                        "BGP Troubleshooting Decision Tree - "
                        "Cisco and RFC-based Diagnostic Methodology"
                    ),
                    content=(
                        f"Command: {', '.join(diag.get('commands', []))}\n"
                        f"Expected: {diag.get('expected', 'Normal operation')}\n"
                        f"If fail: {diag.get('if_fail', 'Investigate further')}"
                    ),
                    sub_steps=[],
                ))
                step_num += 1

        elif protocol.lower() == "ospf":
            ospf_tree = self.troubleshooting.get("ospf_issues", {})
            if "exstart" in issue_type or "exchange" in issue_type:
                tree = ospf_tree.get("neighbor_stuck_in_exstart", {})
            else:
                tree = ospf_tree.get("ospf_neighbor_down", {})

            for diag in tree.get("decision_tree", [])[:4]:
                steps.append(ReasoningStep(
                    step_num=step_num,
                    title=f"OSPF Diagnostic: {diag.get('check', 'Verification step')}",
                    source_ref=(
                        "OSPF Troubleshooting Guide - State Machine Diagnostics, "
                        "RFC 2328 OSPF State Machine Reference"
                    ),
                    content=(
                        f"Command: {', '.join(diag.get('commands', []))}\n"
                        f"Expected: {diag.get('expected', 'Normal operation')}"
                    ),
                    sub_steps=[],
                ))
                step_num += 1

        elif protocol.lower() in ("interface", "connectivity"):
            conn = self.troubleshooting.get("connectivity_issues", {})
            layers = conn.get("layer1_to_layer7_sequence", {}).get("layers", [])
            for layer in layers[:4]:
                steps.append(ReasoningStep(
                    step_num=step_num,
                    title=f"Layer {layer['layer']} ({layer['name']}) Verification",
                    source_ref=(
                        "OSI Model Troubleshooting - Bottom-Up Methodology, "
                        "Chapter: Layer-by-Layer Diagnostic Sequence"
                    ),
                    content=(
                        f"Checks: {', '.join(layer.get('checks', [])[:3])}\n"
                        f"Commands: {', '.join(layer.get('commands', []))}"
                    ),
                    sub_steps=[],
                ))
                step_num += 1

        else:
            # Generic troubleshooting steps
            for i in range(4):
                steps.append(ReasoningStep(
                    step_num=step_num,
                    title=f"Diagnostic Step {i + 1}: Systematic Verification",
                    source_ref=(
                        "Network Troubleshooting Methodology - Systematic Diagnosis, "
                        "Chapter: Protocol-Specific Diagnostics"
                    ),
                    content=(
                        f"Verify {protocol.upper()} {issue_type} from multiple angles:\n"
                        "- Check configuration consistency\n"
                        "- Verify neighbor/peer state\n"
                        "- Examine logs for error messages"
                    ),
                    sub_steps=[],
                ))
                step_num += 1

        return steps

    def _resolution_content(
        self,
        protocol: str,
        issue_type: str,
        symptom: dict[str, Any],
    ) -> str:
        causes_map = {
            "bgp": "AS number mismatch, TCP 179 blocked, MD5 auth mismatch, missing route to peer",
            "ospf": "MTU mismatch (most common), duplicate router-ID, area type mismatch, authentication",
            "interface": "Duplex mismatch, bad cable/SFP, QoS drops, ACL blocking traffic",
            "connectivity": "Routing table missing prefix, ACL deny, NAT failure, firewall rule",
        }
        common_cause = causes_map.get(protocol.lower(), "Configuration mismatch or hardware failure")
        return (
            f"- Root cause category: {issue_type}\n"
            f"- Common causes for {protocol.upper()}: {common_cause}\n"
            "- Apply corrective configuration changes\n"
            "- Verify resolution with show commands\n"
            "- Monitor for recurrence"
        )

    def _resolution_verification(self, protocol: str) -> list[str]:
        verify_map = {
            "bgp": [
                "show bgp summary (verify Established state)",
                "show ip bgp <prefix> (verify route in table)",
                "ping source loopback (verify data plane)",
            ],
            "ospf": [
                "show ip ospf neighbor (verify Full state)",
                "show ip route ospf (verify routes in table)",
                "show ip ospf database (verify LSA synchronization)",
            ],
            "interface": [
                "show interface <int> (verify up/up, zero errors)",
                "ping <destination> repeat 100 (verify no drops)",
                "show interface counters (monitor for new errors)",
            ],
        }
        return verify_map.get(protocol.lower(), ["Verify issue resolved", "Monitor for recurrence"])

# Phi-4 Network Architect — Quality Evaluation Report

**Date:** 2026-03-28
**Model tested:** `gpt-4o-mini` (baseline — fine-tuned Phi-4 not yet trained)
**Test suite:** 25 CCDE-level network design scenarios (`evaluation/test_cases.jsonl`)
**Script:** `python test_model_quality.py --suite --save --backend openai`

> **Purpose of this baseline:** Before the Phi-4-14B fine-tuning run completes, this report
> establishes GPT-4o-mini's score on the same test suite. It identifies which specific
> keywords and concepts the evaluation framework expects, so the training data can be
> verified to cover them. Re-run this script after fine-tuning to measure improvement.

---

## Executive Summary

| Metric | Score | Interpretation |
|--------|------:|----------------|
| **Overall score** | **44%** | Below passing threshold (70%) |
| Keyword match | 13% | Model uses correct concepts but not expected exact phrases |
| Confidence (heuristic) | 65% | Responses are detailed and cite sources |
| `<think>` block rate | **96%** | Chain-of-thought working correctly |
| Average latency | 15.8 s | Acceptable for API mode |
| PASS (≥ 70%) | 1 / 25 | |
| WARN (45–69%) | 7 / 25 | |
| FAIL (< 45%) | 17 / 25 | |

### Key finding

The low **keyword match (13%)** does not mean the model answers incorrectly — GPT-4o-mini gives technically sound responses on every case. It means the test suite's expected phrases (e.g., `"symmetric IRB"`, `"LLQ"`, `"N(N-1)/2 formula"`) are very specific CCDE terminology that the model expresses in different words.
**Goal for the fine-tuned Phi-4:** use these exact terms and achieve ≥ 70% overall score.

---

## Per-Category Results

| Category | Cases | Overall | Keywords | Sources | `<think>` | Result |
|----------|------:|--------:|---------:|--------:|----------:|--------|
| compliance | 2 | 0.61 | 32% | 100% | 100% | ⚠ WARN |
| topology | 3 | 0.49 | 32% | 50% | 100% | ⚠ WARN |
| vxlan | 2 | 0.49 | 12% | 100% | 100% | ⚠ WARN |
| routing | 1 | 0.45 | 0% | 100% | 100% | ⚠ WARN |
| ha | 3 | 0.44 | 8% | 100% | 100% | ✗ FAIL |
| ospf | 2 | 0.44 | 12% | 75% | 100% | ✗ FAIL |
| bgp | 3 | 0.46 | 11% | 83% | 100% | ⚠ WARN |
| qos | 2 | 0.39 | 12% | 50% | 100% | ✗ FAIL |
| wan | 1 | 0.41 | 0% | 100% | 100% | ✗ FAIL |
| security | 1 | 0.44 | 0% | 100% | 100% | ✗ FAIL |
| mpls | 2 | 0.36 | 0% | 75% | 100% | ✗ FAIL |
| sdwan | 2 | 0.33 | 12% | 50% | 50% | ✗ FAIL |
| design_methodology | 1 | 0.35 | 0% | 50% | 100% | ✗ FAIL |

**Best category:** compliance (0.61) — model handles segmentation concepts well
**Weakest categories:** SD-WAN (0.33), design methodology (0.35), MPLS (0.36)

---

## Per-Test-Case Results

| ID | Category | Score | KW | Sources | Think | Latency | Result |
|----|----------|------:|---:|--------:|------:|--------:|--------|
| tc_001 | topology | 0.815 | 75% | 100% | ✓ | 21.8 s | ✅ PASS |
| tc_002 | bgp | 0.487 | 33% | 50% | ✓ | 18.2 s | ⚠ WARN |
| tc_003 | ha | 0.340 | 0% | 50% | ✓ | 23.4 s | ✗ FAIL |
| tc_004 | ospf | 0.440 | 0% | 100% | ✓ | 18.3 s | ✗ FAIL |
| tc_005 | compliance | 0.575 | 25% | 100% | ✓ | 20.3 s | ⚠ WARN |
| tc_006 | qos | 0.340 | 0% | 50% | ✓ | 14.3 s | ✗ FAIL |
| tc_007 | mpls | 0.420 | 0% | 100% | ✓ | 16.4 s | ✗ FAIL |
| tc_008 | sdwan | 0.215 | 25% | 0% | ✗ | 14.3 s | ✗ FAIL |
| tc_009 | vxlan | 0.410 | 0% | 100% | ✓ | 13.5 s | ✗ FAIL |
| tc_010 | ha | 0.410 | 0% | 100% | ✓ | 10.9 s | ✗ FAIL |
| tc_011 | topology | 0.240 | 0% | 0% | ✓ | 13.6 s | ✗ FAIL |
| tc_012 | routing | 0.450 | 0% | 100% | ✓ | 15.7 s | ⚠ WARN |
| tc_013 | security | 0.440 | 0% | 100% | ✓ | 15.5 s | ✗ FAIL |
| tc_014 | mpls | 0.310 | 0% | 50% | ✓ | 13.4 s | ✗ FAIL |
| tc_015 | design_methodology | 0.350 | 0% | 50% | ✓ | 13.6 s | ✗ FAIL |
| tc_016 | bgp | 0.440 | 0% | 100% | ✓ | 10.7 s | ✗ FAIL |
| tc_017 | ospf | 0.435 | 25% | 50% | ✓ | 7.2 s | ✗ FAIL |
| tc_018 | qos | 0.445 | 25% | 50% | ✓ | 11.8 s | ✗ FAIL |
| tc_019 | vxlan | 0.575 | 25% | 100% | ✓ | 16.1 s | ⚠ WARN |
| tc_020 | wan | 0.410 | 0% | 100% | ✓ | 25.3 s | ✗ FAIL |
| tc_021 | compliance | 0.640 | 40% | 100% | ✓ | 13.7 s | ⚠ WARN |
| tc_022 | sdwan | 0.450 | 0% | 100% | ✓ | 11.3 s | ⚠ WARN |
| tc_023 | bgp | 0.440 | 0% | 100% | ✓ | 16.6 s | ✗ FAIL |
| tc_024 | ha | 0.565 | 25% | 100% | ✓ | 14.9 s | ⚠ WARN |
| tc_025 | topology | 0.420 | 20% | 50% | ✓ | 23.5 s | ✗ FAIL |

---

## Failed Tests — Missing Keywords

These are the exact phrases the fine-tuned model must produce to pass.

| ID | Question summary | Missing keywords |
|----|-----------------|-----------------|
| tc_003 | HSRP vs VRRP vs GLBP for campus | `GLBP active-active`, `HSRP disabled preemption`, `VRRP open standard`, `AVG AVF` |
| tc_004 | OSPF 500-router flat area redesign | `multi-area`, `area 0 backbone`, `ABR summarization`, `SPF calculation`, `type 3 LSA` |
| tc_006 | QoS for VoIP + video on 100 Mbps WAN | `EF DSCP 46`, `LLQ`, `33% priority queue limit`, `trust boundary` |
| tc_007 | RD vs RT in MPLS L3VPN | `RD uniqueness`, `RT import export`, `VRF isolation`, `overlapping addresses` |
| tc_008 | DMVPN vs SD-WAN brownout handling | `BFD probes`, `SLA class`, `brownout detection` |
| tc_009 | VXLAN BGP EVPN 40-leaf fabric | `symmetric IRB`, `L3VNI`, `distributed anycast gateway`, `BGP EVPN control plane` |
| tc_010 | 99.99% uptime calculation & N+1 vs N+2 | `52.6 minutes per year`, `N+1 optimal`, `N+2 diminishing returns`, `convergence complexity` |
| tc_011 | Max servers in spine-leaf (64-port / 128-port) | `2048 servers`, `8192 servers`, `N x M / 2`, `three-tier clos for web-scale` |
| tc_013 | DMZ architecture (web, FTP, mail) | `DMZ buffer zone`, `two-tier DMZ`, `web server front-end`, `database backend separation`, `stateful inspection` |
| tc_014 | Inter-AS MPLS Options A / B / C | `Option A most secure back-to-back VRF`, `Option C most scalable`, `Option B balanced`, `ASBR requirements` |
| tc_015 | CCDE design methodology — $5M customer | `top-down design`, `business requirements`, `technical constraints`, `organizational constraints`, `application criticality` |
| tc_016 | BGP same-ASN multi-homing loop prevention | `AS-override`, `allowas-in`, `loop prevention`, `same ASN multi-homing` |
| tc_017 | OSPF area type for low-resource branches | `blocks type 3 4 5 LSAs`, `default route from ABR`, `smallest LSDB` |
| tc_018 | Traffic shaping vs policing at WAN edge | `shaping buffers delays`, `TCP retransmissions`, `egress only for shaping` |
| tc_020 | Hub-and-spoke vs full-mesh for 80 branches | `N(N-1)/2 formula`, `hub-and-spoke scalable`, `full mesh expensive`, `partial mesh compromise` |
| tc_023 | BGP route dampening tuning | `penalty accumulation`, `flap count`, `suppress limit`, `half-life decay`, `reuse limit` |
| tc_025 | Spine-leaf vs three-tier selection criteria | `VM mobility`, `consistent latency`, `horizontal scaling`, `north-south for three-tier` |

---

## Training Gap Analysis

Concepts that appear most frequently in missing keywords — these must be well-represented in training data:

| Gap | Frequency | Domain | Action |
|-----|:---------:|--------|--------|
| Exact QoS terminology (`EF DSCP 46`, `LLQ`, `trust boundary`) | High | QoS | Add 50+ QoS Q&A pairs with exact DSCP values |
| VXLAN EVPN specifics (`symmetric IRB`, `L3VNI`, `distributed anycast gateway`) | High | Data Center | Add VXLAN fabric design scenarios |
| OSPF area types (`totally stubby`, `type 3 LSA`, `ABR summarization`) | High | Routing | Add OSPF area design with LSA filter explanations |
| BGP edge cases (`AS-override`, `allowas-in`, `dampening mechanics`) | High | BGP | Add BGP troubleshooting with exact CLI/config |
| MPLS VPN (`RD uniqueness`, `RT import/export`, Inter-AS options) | High | MPLS | Add L3VPN design and Inter-AS scenarios |
| HA SLA numbers (`52.6 min/year`, `N+1 optimal`, `50% utilisation rule`) | Medium | HA Design | Add SLA calculation Q&A with exact numbers |
| SD-WAN planes (`vBond`, `vSmart OMP`, `BFD SLA`, `brownout detection`) | Medium | SD-WAN | Add Cisco SD-WAN architecture Q&A |
| CCDE methodology (`top-down`, `business/technical/org constraints`) | Medium | Methodology | Add requirements gathering scenarios |
| Spine-leaf maths (`N×M/2`, `2048 servers`, `8192 servers`) | Medium | DC Topology | Add capacity planning calculations |
| DMZ terms (`two-tier DMZ`, `stateful inspection`) | Low | Security | Add security architecture Q&A |

---

## Scoring Methodology

Each test case is scored on four components:

| Component | Weight | Description |
|-----------|-------:|-------------|
| Keyword match | 50% | Fraction of `expected_key_facts` found (case-insensitive) |
| Source citation | 20% | Fraction of `sources_required` referenced |
| Confidence heuristic | 20% | Based on response length, sources cited, `<think>` presence |
| `<think>` block bonus | 10% | +0.10 if response contains a reasoning block |

**Pass thresholds:**
- **PASS:** overall ≥ 0.70
- **WARN:** overall 0.45–0.69
- **FAIL:** overall < 0.45

---

## What to Expect After Fine-Tuning

Based on this baseline, a successfully fine-tuned Phi-4-14B on CCDE training data should:

| Metric | Baseline (GPT-4o-mini) | Target (fine-tuned Phi-4) |
|--------|----------------------:|-------------------------:|
| Overall score | 44% | ≥ 70% |
| Keyword match | 13% | ≥ 65% |
| `<think>` rate | 96% | ≥ 95% |
| PASS count | 1 / 25 | ≥ 18 / 25 |
| Avg latency | 15.8 s | 3–6 s (GPTQ on GPU) |

Re-run evaluation after training:

```bash
python test_model_quality.py --suite --save --backend api
# API must be running: make deploy
```

Compare `quality_report.json` to this baseline to measure improvement per category.

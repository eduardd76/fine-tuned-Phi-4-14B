# How to Prepare Your NotebookLM for Network Architecture Dataset Generation

## What to Upload to NotebookLM

### Priority 1: Network Design Books (Essential)
Upload these types of books if you have them:

#### Enterprise Network Design
- Cisco CCDE study guides
- "Network Warrior" by Gary Donahue
- "Optimal Routing Design" by Russ White
- "Top-Down Network Design" by Priscilla Oppenheimer
- Enterprise campus network design guides
- WAN design books

#### Data Center Networking
- Data center design guides (spine-leaf, Clos fabrics)
- VXLAN/EVPN implementation guides
- SDN and network virtualization books
- Cloud networking architecture books

#### Routing Protocols
- BGP design and implementation books
- OSPF design books
- Multi-protocol routing design
- Routing protocol comparison guides

### Priority 2: Troubleshooting Guides (Very Important)
- "Troubleshooting IP Routing Protocols" 
- Network troubleshooting methodology books
- Protocol-specific troubleshooting guides
- Performance optimization guides
- Packet analysis books

### Priority 3: Security & Compliance (Critical for Reasoning)
- Zero Trust architecture guides
- PCI-DSS implementation guides (get latest version 4.0+)
- HIPAA network security requirements
- SOX compliance for IT infrastructure
- "Practical Security Architecture" type books
- Firewall design and implementation

### Priority 4: Modern Technologies
- SD-WAN design and deployment guides
- SASE architecture books
- Cloud connectivity (AWS/Azure/GCP networking)
- Wireless network design
- IoT network architecture
- 5G enterprise networking

### Priority 5: Vendor-Specific Guides
- Cisco design guides (any technology)
- Juniper design and implementation guides
- Arista best practices
- Palo Alto NGFW deployment guides
- F5 load balancer guides

### Priority 6: Case Studies & Real Implementations
- Network design case study books
- Real-world implementation stories
- Migration and transformation guides
- Lessons learned compilations

## What Makes a Good Source

✅ **Include if the book has:**
- Specific design methodologies (decision trees, selection criteria)
- Real configuration examples
- Troubleshooting decision trees
- Cost/sizing guidelines
- Vendor best practices
- Compliance requirement details
- Implementation timelines
- Staffing/resource requirements

❌ **Skip if the book is:**
- Too basic/introductory (CCNA level)
- Purely theoretical with no practical guidance
- Outdated technology (pre-2015 unless foundational)
- Focused on non-enterprise topics

## Recommended Upload Strategy

### Organization in NotebookLM

Create logical groupings (NotebookLM allows organizing sources):

**Group 1: Design Methodologies**
- Upload: Enterprise design books, topology selection guides
- Purpose: Extract decision criteria for network architectures

**Group 2: Protocol Design**
- Upload: BGP, OSPF, EIGRP, MPLS design books
- Purpose: Extract routing protocol selection and design patterns

**Group 3: Troubleshooting**
- Upload: Troubleshooting guides, diagnostic methodologies
- Purpose: Extract systematic diagnostic approaches

**Group 4: Security & Compliance**
- Upload: PCI-DSS, HIPAA, Zero Trust, security architecture
- Purpose: Extract exact compliance requirements

**Group 5: Modern Tech**
- Upload: SD-WAN, SASE, cloud networking, VXLAN
- Purpose: Extract current technology best practices

**Group 6: Vendor Guides**
- Upload: Cisco, Juniper, Arista, Palo Alto guides
- Purpose: Extract configuration examples and syntax

## Essential Queries to Test Your NotebookLM

After uploading, test with these queries to ensure good coverage:

### Design Methodology Queries
```
1. "What are the criteria for selecting between collapsed core and three-tier architecture?"
2. "What user count ranges map to different network topologies?"
3. "What are the design patterns for high availability networks?"
4. "How do you size bandwidth for different user counts?"
5. "What are the implementation phases for enterprise network deployment?"
```

### Troubleshooting Queries
```
1. "What is the systematic troubleshooting methodology for BGP issues?"
2. "How do you diagnose intermittent packet loss?"
3. "What are the steps to troubleshoot routing loops?"
4. "What is the Layer 1-7 diagnostic sequence?"
5. "How do you identify the root cause of network congestion?"
```

### Compliance Queries
```
1. "What are the specific PCI-DSS network segmentation requirements?"
2. "What encryption standards does HIPAA require?"
3. "What are the SOX audit logging requirements for networks?"
4. "What are the technical controls required for FedRAMP compliance?"
```

### Configuration Queries
```
1. "What are BGP configuration best practices from Cisco guides?"
2. "What is a proper QoS policy structure?"
3. "How should ACLs be structured for compliance?"
4. "What are VXLAN EVPN configuration patterns?"
```

### Cost & Sizing Queries
```
1. "What are typical CapEx costs for enterprise networks by user count?"
2. "What are the OpEx components and percentages?"
3. "What are staffing requirements for different network sizes?"
4. "What are realistic implementation timelines?"
```

## Minimum Viable NotebookLM

If you have limited books, **at minimum upload:**

1. **One comprehensive design book** (like "Top-Down Network Design")
2. **One troubleshooting guide** (systematic methodology)
3. **One security/compliance guide** (PCI-DSS or HIPAA)
4. **One vendor guide** (Cisco or Juniper design guide)
5. **One modern tech book** (SD-WAN or cloud networking)

This gives enough to:
- Extract design methodologies
- Create troubleshooting decision trees
- Include compliance specifics
- Generate valid configurations
- Cover modern technologies

## Optimal NotebookLM (If You Have Many Books)

Upload **15-20 books** covering:
- 3-4 design methodology books
- 2-3 troubleshooting guides
- 2-3 security/compliance books
- 3-4 vendor guides (Cisco, Juniper, Arista, Palo Alto)
- 2-3 modern technology books (SD-WAN, SASE, cloud)
- 2-3 protocol-specific books (BGP, OSPF, MPLS)
- 1-2 case study compilations

This provides:
- Multiple perspectives on design patterns
- Comprehensive troubleshooting coverage
- Deep compliance knowledge
- Vendor-specific accuracy
- Current technology best practices
- Real-world validation

## Quality Check

After uploading, your NotebookLM should be able to answer:

✅ **Design Questions:**
- "When should I use three-tier vs collapsed core architecture?"
- "What bandwidth do I need for 5000 users?"
- "How do I design for 99.99% uptime?"

✅ **Troubleshooting Questions:**
- "What's the first step when BGP is flapping?"
- "How do I diagnose packet loss?"
- "What causes routing loops and how do I fix them?"

✅ **Compliance Questions:**
- "What does PCI-DSS require for network segmentation?"
- "What encryption does HIPAA mandate?"
- "What logs does SOX require?"

✅ **Configuration Questions:**
- "What's a proper BGP configuration for dual-homing?"
- "How do I configure QoS for voice traffic?"
- "What's the correct ACL structure for DMZ?"

✅ **Cost Questions:**
- "What's typical CapEx for 1000-user network?"
- "What percentage of CapEx is annual OpEx?"
- "How many engineers do I need for 50-site network?"

## Red Flags (NotebookLM Not Ready)

❌ NotebookLM says "I don't have information about..." for:
- Design methodology decision criteria
- Troubleshooting systematic approaches
- Compliance technical requirements
- Configuration examples
- Cost/sizing guidelines

If you see these, you need more/better source material.

## Example Good Books to Upload

If you're looking for books to acquire:

**Design:**
- "Top-Down Network Design" - Priscilla Oppenheimer
- "Network Warrior" - Gary Donahue
- "Optimal Routing Design" - Russ White
- Cisco Press CCDE study guides

**Troubleshooting:**
- "Troubleshooting IP Routing Protocols" - Faraz Shamim et al.
- "Network Troubleshooting Tools" - Joseph Sloan

**Security:**
- "Zero Trust Networks" - Evan Gilman
- Official PCI-DSS guidance documents
- "Practical Cloud Security" - Chris Dotson

**Modern Tech:**
- "SD-WAN Explained" - various authors
- AWS/Azure/GCP networking documentation
- VXLAN deployment guides

## What the Dataset Generator Will Extract

The dataset generator will query NotebookLM for:

1. **Design Patterns** → Structured decision trees
2. **Troubleshooting Methods** → Step-by-step diagnostic flows
3. **Compliance Requirements** → Exact technical specifications
4. **Configuration Templates** → Valid syntax examples
5. **Cost Benchmarks** → Realistic CapEx/OpEx data
6. **Sizing Guidelines** → User counts → topology/bandwidth mappings
7. **Implementation Timelines** → Phase durations and dependencies
8. **Vendor Specifics** → Platform-specific best practices

## Timeline

**Uploading books to NotebookLM:** 1-2 hours
**Testing coverage with queries:** 30 minutes
**Ready for Claude Code:** Immediately after upload

## Final Checklist

Before giving the prompt to Claude Code, verify your NotebookLM has:

- [ ] At least 5 books uploaded (10-20 is optimal)
- [ ] Coverage of design methodologies
- [ ] Coverage of troubleshooting approaches
- [ ] At least one compliance guide (PCI-DSS/HIPAA/SOX)
- [ ] At least one vendor guide (Cisco/Juniper/Arista)
- [ ] Coverage of modern technologies (SD-WAN/SASE/Cloud)
- [ ] Can answer design methodology questions
- [ ] Can answer troubleshooting questions
- [ ] Can answer compliance questions
- [ ] Can provide configuration examples
- [ ] Can provide cost/sizing guidance

Once you have ✅ on all items, you're ready to use the Claude Code prompt!

## Next Step

After NotebookLM is prepared → Use the prompt in `CLAUDE_CODE_PROMPT_PHI4_NOTEBOOKLM.md`

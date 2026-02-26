# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

Report security vulnerabilities to **security@aumos.ai**.

**Do not open a public GitHub issue for security vulnerabilities.**

Include:
1. Description of the vulnerability
2. Steps to reproduce
3. Potential impact assessment
4. Suggested remediation (if any)

We respond within 48 hours and aim to patch critical issues within 7 days.

## Security Considerations

- All API endpoints require authentication via aumos-auth-gateway (JWT)
- Tenant isolation enforced via PostgreSQL Row Level Security (RLS)
- Compliance gates are immutable once triggered — they cannot be bypassed by confidence scores
- HITL review decisions are immutable — once submitted they cannot be overwritten
- Feedback corrections never directly update AI model weights — they only trigger calibration recommendations requiring human approval
- All confidence scores and routing decisions are auditable via hac_routing_decisions table

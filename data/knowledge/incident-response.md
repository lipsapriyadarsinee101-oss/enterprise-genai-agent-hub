# AI Incident Response Playbook

## Triage
Potential incidents are classified by user impact, data exposure, regulatory risk, and service disruption. Critical incidents are escalated immediately to the service owner, security team, and responsible business owner.

## Containment
The team can disable generation, switch to a safe fallback, revoke a data source, or roll back the model and index. Evidence including request IDs, model versions, retrieved chunks, and relevant logs must be preserved.

## Recovery and learning
Service is restored only after the owner validates the fix and completes regression tests. A post-incident review records root cause, impact, corrective actions, owners, and deadlines.

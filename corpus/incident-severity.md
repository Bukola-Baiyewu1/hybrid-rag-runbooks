# Incident Severity and Communication

## Severity levels
- SEV1: customer-facing outage or data loss. Page the incident commander
  immediately.
- SEV2: major feature degraded for many customers, or a SEV1 risk.
- SEV3: minor degradation with a workaround.

## Communication
For SEV1 and SEV2, post an update on the status page within 15 minutes of
declaring the incident and then every 30 minutes until resolved. Use the
incident Slack channel named `#inc-<date>-<short-name>` for all coordination.

## After the incident
Every SEV1 and SEV2 gets a blameless postmortem within 5 working days, with
action items that have an owner and a due date.

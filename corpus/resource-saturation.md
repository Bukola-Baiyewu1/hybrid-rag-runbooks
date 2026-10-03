# CPU or Memory Saturation and High Latency

## Symptoms
CPU stays above 85% or p95 latency stays above its target for 10 minutes.
Requests queue up and clients see slow responses rather than errors.

## Diagnosis
Confirm the load is real and not a metrics glitch by checking current CPU,
replica count, and error rate (`get_service_health`). Compare request volume
with the same hour last week to tell organic growth from a traffic spike.

## Remediation
Scale out by exactly one replica to shed load (`scale_service`). Never add more
than one replica per approval and never exceed the configured replica limit,
which is 5 replicas for the web service.

## Escalation
If saturation or high latency persists 10 minutes after scaling, escalate to a
human. Persistent saturation usually means a code regression or a slow
database query rather than missing capacity.

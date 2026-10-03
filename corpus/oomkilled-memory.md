# Container OOMKilled

## Symptoms
`kubectl describe pod` shows `Last State: Terminated, Reason: OOMKilled` and
exit code 137. Memory usage climbs steadily until the container is killed.

## Diagnosis
Compare the container's memory usage with `resources.limits.memory` in the pod
spec. A steady climb that never levels off suggests a memory leak; a sudden
jump suggests a large request or batch job.

## Remediation
As a short-term fix, raise `resources.limits.memory` by at most 25% and set
`resources.requests.memory` to the observed steady-state usage. For a suspected
leak, capture a heap profile before the next restart and open a bug for the
owning team. For JVM services, keep `-Xmx` at about 75% of the memory limit so
the JVM has headroom for non-heap memory.

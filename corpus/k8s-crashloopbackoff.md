# Pod in CrashLoopBackOff

## Symptoms
`kubectl get pods` shows a pod in `CrashLoopBackOff` with a rising RESTARTS
count. Kubernetes waits longer between each restart, up to five minutes.

## Diagnosis
Read the logs of the crashed container, not the new one:
`kubectl logs <pod> --previous`. Then run `kubectl describe pod <pod>` and read
the Last State section. Exit code 137 means the container was killed, usually
by the out-of-memory killer (see the oomkilled-memory runbook). Exit code 1
usually means the application failed on startup, often because of a missing
environment variable or secret.

## Common causes
- A failing liveness probe that kills a slow-starting container. Increase
  `initialDelaySeconds` or add a `startupProbe`.
- A missing ConfigMap or Secret referenced in the pod spec.
- A bad image tag after a deploy; check the image with `kubectl describe pod`.

## Remediation
Fix the root cause rather than deleting the pod; a deleted pod is recreated and
crashes again. If the crash started with a deploy, roll back the deploy.

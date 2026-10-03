# Kubernetes Node NotReady

## Symptoms
`kubectl get nodes` shows a node as `NotReady`. Pods on the node may be stuck
in Terminating or Unknown, and new pods are not scheduled there.

## Diagnosis
Run `kubectl describe node <node>` and read the Conditions section for
MemoryPressure, DiskPressure, or PIDPressure. On the node itself, check the
kubelet with `systemctl status kubelet` and `journalctl -u kubelet`.

## Remediation
Stop new pods from landing on the node with `kubectl cordon <node>`, then move
workloads off it with `kubectl drain <node> --ignore-daemonsets --delete-emptydir-data`.
If the kubelet does not recover after a restart, replace the node through the
node pool rather than repairing it by hand.

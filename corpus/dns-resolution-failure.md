# DNS Resolution Failures in the Cluster

## Symptoms
Applications log `getaddrinfo ENOTFOUND` or `NXDOMAIN` for internal service
names, or DNS lookups are slow and time out after 5 seconds.

## Diagnosis
Check that CoreDNS pods are running: `kubectl -n kube-system get pods -l k8s-app=kube-dns`.
Test resolution from inside a pod with `nslookup <service>.<namespace>.svc.cluster.local`.
The default `ndots:5` setting makes every external name try five search
domains first, which multiplies DNS queries.

## Remediation
If CoreDNS pods are crash-looping or overloaded, scale the CoreDNS deployment
to at least 3 replicas. To reduce query volume for external names, set
`dnsConfig.options` with `ndots: 2` in the pod spec, or use fully qualified
names ending in a dot.

# Disk Pressure

## Symptoms
Disk usage on a host rises above 85%. The node may report DiskPressure and
begin evicting pods.

## Diagnosis
Identify the largest directories with `du -xh --max-depth=1 / | sort -h`.
Common culprits are unrotated log files, orphaned container images, and stale
build caches. Check `docker system df` on build agents.

## Remediation
Disk issues are never auto-remediated because deleting data is destructive.
Always escalate disk pressure to a human, who can safely clear logs, prune
images with `docker image prune`, or expand the volume.

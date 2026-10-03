# Redis Memory Pressure

## Symptoms
Redis `used_memory` approaches `maxmemory`, the `evicted_keys` counter rises,
or writes fail with `OOM command not allowed when used memory > 'maxmemory'`.

## Diagnosis
Run `INFO memory` and `INFO stats`. Check the eviction policy with
`CONFIG GET maxmemory-policy`. The default `noeviction` policy rejects writes
when memory is full instead of evicting keys.

## Remediation
For a cache, set `maxmemory-policy allkeys-lru` so the least recently used keys
are evicted. For data that must not be lost, keep `noeviction` and increase
`maxmemory` instead. Find large keys with `redis-cli --bigkeys`, and make sure
cache keys have a TTL.

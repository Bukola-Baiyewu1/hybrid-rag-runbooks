# Nginx 502 Bad Gateway

## Symptoms
Clients receive `502 Bad Gateway` from the Nginx reverse proxy. The Nginx error
log shows `upstream prematurely closed connection` or `connect() failed (111:
Connection refused) while connecting to upstream`.

## Diagnosis
`Connection refused` means the upstream application is down or not listening on
the expected port. `upstream prematurely closed connection` usually means the
upstream timed out or crashed mid-request. `504 Gateway Timeout` instead of 502
means Nginx gave up waiting, controlled by `proxy_read_timeout` (default 60s).

## Remediation
Confirm the upstream service is healthy first; most 502s are upstream problems,
not Nginx problems. Only raise `proxy_read_timeout` for endpoints that are
legitimately slow, such as report exports. Enable upstream keepalive with
`keepalive 32;` in the upstream block to reduce connection churn.

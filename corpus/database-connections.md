# Database Connection Pool Exhausted

## Symptoms
The application logs `remaining connection slots are reserved for non-replication
superuser connections` or `FATAL: sorry, too many clients already`, and requests
time out waiting for a database connection.

## Diagnosis
Count connections per application with
`SELECT usename, application_name, count(*) FROM pg_stat_activity GROUP BY 1, 2;`.
Look for idle-in-transaction sessions, which hold connections without doing work.

## Remediation
Database restarts, failovers, and changes to `max_connections` are never
automated. The application pool size is set by `DB_POOL_SIZE` (default 10 per
replica); scaling out the application multiplies connections, so do not scale
out to fix pool exhaustion.

## Escalation
Escalate to the database owner with the pg_stat_activity output and the error
messages from the recent logs.

# Message Queue Backlog

## Symptoms
The RabbitMQ queue depth keeps growing and consumer lag rises. Jobs that
normally finish in seconds take minutes.

## Diagnosis
Compare the publish rate with the consumer acknowledge rate in the RabbitMQ
management UI. If consumers are connected but slow, check `prefetch_count`: a
value that is too high lets one consumer hoard messages while others sit idle.
A value of 10 to 50 suits most workers.

## Remediation
Add consumers by scaling the worker deployment. Messages that fail repeatedly
are moved to the dead-letter queue `jobs.dlq` after 5 attempts; inspect them
before replaying, because replaying poison messages refills the backlog.

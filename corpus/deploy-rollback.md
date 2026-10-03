# Rolling Back a Bad Deploy

## When to roll back
Roll back when a deploy causes errors, crashes, or latency regressions and a
fix cannot be shipped within 30 minutes. Rolling back is preferred over
debugging in production during an incident.

## Kubernetes deployments
Check the history with `kubectl rollout history deployment/<name>`, then roll
back to the previous revision with `kubectl rollout undo deployment/<name>`.
Watch progress with `kubectl rollout status deployment/<name>`.

## Helm releases
List revisions with `helm history <release>` and roll back with
`helm rollback <release> <revision>`.

## Database migrations
A rollback does not undo database migrations. If the bad deploy ran a
migration, check with the owning team before rolling back, because old code may
not work with the new schema.

# High Error Rate (5xx)

## Symptoms
The service returns HTTP 5xx responses above 5% for more than 5 minutes.
Users report failed requests and timeouts. The `http_requests_total{code=~"5.."}`
panel on the service dashboard turns red.

## Diagnosis
Check recent logs for stack traces or unhandled exceptions (`get_recent_logs`).
Look at whether the error spike lines up with a recent deploy by comparing the
error start time to the deploy timestamp in the release history.

## Remediation
If the errors began shortly after a deploy, restart the service to clear the
faulty process (`restart_service`). This is the fastest safe first response.
If restarting does not help and CPU or memory is saturated, scale out by
exactly one replica (`scale_service`).

## Escalation
If neither applies, or errors continue 15 minutes after a restart, do not act
further. Escalate to the human on-call engineer and consider a rollback using
the deploy-rollback runbook.

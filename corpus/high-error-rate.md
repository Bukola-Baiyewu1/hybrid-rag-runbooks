# High Error Rate (5xx)

## Symptoms
The service returns HTTP 5xx responses above 5% for more than 5 minutes.
Users report failed requests and timeouts.

## Diagnosis
Check recent logs for stack traces or unhandled exceptions. Look at whether the
error spike lines up with a recent deploy by comparing the error start time to
the deploy timestamp.

## Remediation
If the errors began shortly after a deploy, restart the service to clear the
faulty process. This is the fastest safe first response. If restarting does not
help and CPU or memory is saturated, scale out by one replica. If neither
applies, escalate to a human on-call engineer.

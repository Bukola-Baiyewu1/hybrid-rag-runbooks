# Log Volume Spike

## Symptoms
Log ingestion volume doubles or more within an hour, the logging bill alert
fires, or the log pipeline starts dropping messages.

## Diagnosis
Find the noisiest services by log line count over the last hour. The most
common cause is debug logging left on after an investigation: check whether
`LOG_LEVEL` is set to `DEBUG` in the service configuration.

## Remediation
Set `LOG_LEVEL` back to `INFO` and redeploy. For a service that logs the same
error in a tight loop, fix the loop rather than raising ingestion limits. Never
log request bodies, tokens, or personal data, even at debug level.

# Rotating Application Secrets

## When to rotate
Rotate a secret immediately if it was committed to Git, pasted into a chat or
ticket, or exposed in logs. Rotate database passwords every 90 days.

## Dual-credential rotation
To rotate without downtime, create the new credential while the old one still
works, update the secret in the key vault, roll out the application so every
replica uses the new value, and only then revoke the old credential.

## After rotation
Confirm the old credential is revoked by attempting to use it. Check the
application logs for authentication failures for 30 minutes after the rollout.
Removing a leaked secret from Git history does not make it safe; it must still
be rotated.

# TLS Certificate Expiry Alert

## Trigger
A TLS certificate expires in fewer than 14 days, or has already expired.
Expired certificates cause handshake failures such as
`SSL_ERROR_EXPIRED_CERT_ALERT` in browsers.

## Response
Certificate renewal changes security configuration and must be performed by a
human. Do not restart or scale the service for a certificate alert. Escalate to
the platform team with the certificate subject and expiry date, and follow the
tls-cert-rotation runbook.

# TLS Certificate Rotation

## When to rotate
Rotate the TLS certificate before its expiry date, or immediately if a key
compromise is suspected. The certificate expiry alert fires 14 days before
expiry.

## Steps
1. Generate a new certificate signing request with the existing private key.
2. Submit the CSR to the certificate authority and obtain the signed cert.
3. Install the new certificate into the ingress secret named `tls-web`.
4. Reload the ingress controller so it picks up the new certificate.
5. Verify the new expiry date with `openssl x509 -enddate -noout -in cert.pem`.

## Rollback
If clients report handshake failures after rotation, restore the previous
certificate from the backup secret `tls-web-previous` and reload the ingress.

## Ownership
Certificate renewal changes security configuration, so it is performed by the
platform team and is never automated by the triage agent.

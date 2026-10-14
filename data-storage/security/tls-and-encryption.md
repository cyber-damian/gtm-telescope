# Transport security and encryption

## Encryption in transit

All customer-facing endpoints require TLS 1.3. TLS 1.2 is accepted only for legacy API clients until the end of the deprecation window, and TLS 1.0 and 1.1 are rejected. Public API traffic terminates on port 443; the private connector gateway listens on port 8443 with mutual TLS.

## Encryption at rest

Customer data is encrypted at rest with AES-256. Keys are managed in a cloud KMS and rotated every ninety days. Backups use the same encryption and separate keys.

## Certificates

Certificates are issued by a public CA and renewed automatically thirty days before expiry.

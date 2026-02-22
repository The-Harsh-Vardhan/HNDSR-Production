# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in HNDSR, please report it responsibly:

1. **Do NOT** open a public GitHub issue for security vulnerabilities.
2. Email the maintainer directly or use GitHub's private vulnerability reporting feature.
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will acknowledge receipt within **48 hours** and aim to provide a fix or mitigation within **7 days** for critical issues.

## Security Measures in Place

- **Non-root Docker containers**: Both production and HF Spaces Dockerfiles run as non-root users.
- **Input validation**: Image dimensions capped at 16M pixels, payload size limited to 20 MB.
- **Rate limiting**: Per-IP hourly rate limits to prevent abuse.
- **CORS policy**: Configurable origin allowlist.
- **No secrets in code**: All sensitive configuration via environment variables.
- **Dependency pinning**: Production dependencies pinned in `requirements-prod.txt`.

## Best Practices for Deployment

- Always use HTTPS in production (TLS termination via Nginx or cloud LB).
- Rotate any API keys or tokens regularly.
- Keep dependencies updated — run `pip audit` periodically.
- Monitor the `/metrics` endpoint for anomalous request patterns.

# Technical Documentation

This document explains our API endpoints.

## Authentication

Use the following endpoint to authenticate:
https://api.example.com/auth

## Code Example

```python
def authenticate(token):
    # Send authentication request
    response = api.post("/auth", headers={"Token": token})
    return response.json()
```

## Important Notes

- Always use HTTPS
- Tokens expire after 24 hours
- Rate limit: 100 requests/minute

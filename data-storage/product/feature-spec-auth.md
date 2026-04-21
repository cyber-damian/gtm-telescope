# Feature spec: passwordless sign-in

## Goal

Let returning users sign in with a magic link emailed to their verified address, reducing password fatigue and support tickets for resets.

## Security

Links expire after fifteen minutes and are single-use. Rate-limit requests per email and per IP. Log authentication outcomes for fraud review without storing raw tokens.

## Rollout

Ship behind a feature flag for internal dogfood first, then 5% of new signups before general availability.
